// mtl_dump.m — DYLD_INSERT_LIBRARIES shim that records every Metal shader library compiled
// from source (Dawn/Tint emits MSL text at pipeline-creation time) plus compute pipeline
// creation and dispatch geometry. Output dir: $MTL_DUMP_DIR (default /tmp/mtl_dump).
//   clang -fobjc-arc -dynamiclib -framework Metal -framework Foundation mtl_dump.m -o libmtl_dump.dylib
//   MTL_DUMP_DIR=... DYLD_INSERT_LIBRARIES=libmtl_dump.dylib python ...
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

static NSString *g_dir;
static FILE *g_log;
static int g_libcount = 0;
static int g_dispatch_limit = 4000;   // stop logging dispatches after this many (keeps files small)
static int g_dispatch_count = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
// ablation: skip dispatches whose pipeline came from one of MTL_DUMP_SKIP_LIBS (comma list of
// LIB indices) once the total dispatch count exceeds MTL_DUMP_SKIP_AFTER
#define MAXPSO 4096
static const void *g_pso_ptr[MAXPSO]; static int g_pso_lib[MAXPSO]; static int g_npso = 0;
static int g_skip_libs[256]; static int g_nskip = 0;
static unsigned long long g_skip_hashes[256]; static int g_nskiph = 0;
static unsigned long long g_lib_hash[MAXPSO];   // FNV-1a of the MSL source, by lib index
static unsigned long long fnv1a(const char *s) { unsigned long long h = 0xcbf29ce484222325ULL; for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; } return h; } static long g_skip_after = -1; static long g_total_disp = 0; static long g_skipped = 0;
static void remember_pso(const void *p) { pthread_mutex_lock(&g_mu); if (g_npso < MAXPSO) { g_pso_ptr[g_npso] = p; g_pso_lib[g_npso] = g_libcount - 1; g_npso++; } pthread_mutex_unlock(&g_mu); }
static int lib_of_pso(const void *p) { for (int i = g_npso - 1; i >= 0; i--) if (g_pso_ptr[i] == p) return g_pso_lib[i]; return -1; }
static int should_skip(const void *pso) {
  if (g_skip_after < 0) return 0;
  long n = __sync_add_and_fetch(&g_total_disp, 1);
  if (n <= g_skip_after) return 0;
  int lib = lib_of_pso(pso);
  for (int i = 0; i < g_nskip; i++) if (g_skip_libs[i] == lib) { __sync_add_and_fetch(&g_skipped, 1); return 1; }
  if (lib >= 0 && lib < MAXPSO) for (int i = 0; i < g_nskiph; i++) if (g_skip_hashes[i] == g_lib_hash[lib]) { __sync_add_and_fetch(&g_skipped, 1); return 1; }
  return 0;
}

static void logf_(const char *fmt, ...) {
  if (!g_log) return;
  va_list ap; va_start(ap, fmt);
  pthread_mutex_lock(&g_mu);
  vfprintf(g_log, fmt, ap); fputc('\n', g_log); fflush(g_log);
  pthread_mutex_unlock(&g_mu);
  va_end(ap);
}

// ---- MTLDevice newLibraryWithSource:options:error: ----
typedef id<MTLLibrary> (*NewLibSrcIMP)(id, SEL, NSString *, MTLCompileOptions *, NSError **);
static NewLibSrcIMP orig_newLibSrc;
static id<MTLLibrary> my_newLibSrc(id self, SEL _cmd, NSString *src, MTLCompileOptions *opts, NSError **err) {
  id<MTLLibrary> lib = orig_newLibSrc(self, _cmd, src, opts, err);
  int idx;
  pthread_mutex_lock(&g_mu); idx = g_libcount++; pthread_mutex_unlock(&g_mu);
  NSString *path = [g_dir stringByAppendingFormat:@"/lib_%04d.metal", idx];
  [src writeToFile:path atomically:NO encoding:NSUTF8StringEncoding error:nil];
  NSArray *names = lib ? [lib functionNames] : @[];
  unsigned long long h = fnv1a([src UTF8String]); if (idx < MAXPSO) g_lib_hash[idx] = h;
  logf_("LIB %d bytes=%lu ok=%d hash=%016llx fns=%s", idx, (unsigned long)[src length], lib != nil, h,
        [[names componentsJoinedByString:@","] UTF8String]);
  return lib;
}

// ---- compute pipeline creation (function name -> pipeline pointer) ----
typedef id<MTLComputePipelineState> (*NewCPSFnIMP)(id, SEL, id<MTLFunction>, NSError **);
static NewCPSFnIMP orig_newCPSFn;
static id<MTLComputePipelineState> my_newCPSFn(id self, SEL _cmd, id<MTLFunction> fn, NSError **err) {
  id<MTLComputePipelineState> p = orig_newCPSFn(self, _cmd, fn, err);
  remember_pso((__bridge const void *)p);
  logf_("PSO %p fn=%s maxtg=%lu simd=%lu", p, [[fn name] UTF8String],
        (unsigned long)(p ? p.maxTotalThreadsPerThreadgroup : 0), (unsigned long)(p ? p.threadExecutionWidth : 0));
  return p;
}
typedef id<MTLComputePipelineState> (*NewCPSDescIMP)(id, SEL, MTLComputePipelineDescriptor *, MTLPipelineOption, MTLComputePipelineReflection **, NSError **);
static NewCPSDescIMP orig_newCPSDesc;
static id<MTLComputePipelineState> my_newCPSDesc(id self, SEL _cmd, MTLComputePipelineDescriptor *d, MTLPipelineOption o, MTLComputePipelineReflection **r, NSError **err) {
  id<MTLComputePipelineState> p = orig_newCPSDesc(self, _cmd, d, o, r, err);
  remember_pso((__bridge const void *)p);
  logf_("PSO %p fn=%s label=%s maxtg=%lu simd=%lu", p, [[d.computeFunction name] UTF8String], [d.label UTF8String] ?: "",
        (unsigned long)(p ? p.maxTotalThreadsPerThreadgroup : 0), (unsigned long)(p ? p.threadExecutionWidth : 0));
  return p;
}

// ---- compute encoder: setComputePipelineState / dispatch ----
static __thread const void *t_cur_pso;
typedef void (*SetPSOIMP)(id, SEL, id<MTLComputePipelineState>);
static SetPSOIMP orig_setPSO;
static void my_setPSO(id self, SEL _cmd, id<MTLComputePipelineState> p) { t_cur_pso = (__bridge const void *)p; orig_setPSO(self, _cmd, p); }
typedef void (*DispTGIMP)(id, SEL, MTLSize, MTLSize);
static DispTGIMP orig_dispTG, orig_dispThreads;
static void my_dispTG(id self, SEL _cmd, MTLSize tg, MTLSize tpg) {
  if (should_skip(t_cur_pso)) return;
  if (g_dispatch_count < g_dispatch_limit) { g_dispatch_count++;
    logf_("DISPATCH tg %p grid=(%lu,%lu,%lu) tpg=(%lu,%lu,%lu)", t_cur_pso, (unsigned long)tg.width, (unsigned long)tg.height, (unsigned long)tg.depth, (unsigned long)tpg.width, (unsigned long)tpg.height, (unsigned long)tpg.depth); }
  orig_dispTG(self, _cmd, tg, tpg);
}
static void my_dispThreads(id self, SEL _cmd, MTLSize t, MTLSize tpg) {
  if (should_skip(t_cur_pso)) return;
  if (g_dispatch_count < g_dispatch_limit) { g_dispatch_count++;
    logf_("DISPATCH th %p threads=(%lu,%lu,%lu) tpg=(%lu,%lu,%lu)", t_cur_pso, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth, (unsigned long)tpg.width, (unsigned long)tpg.height, (unsigned long)tpg.depth); }
  orig_dispThreads(self, _cmd, t, tpg);
}

static IMP swz(Class c, SEL sel, IMP imp) {
  Method m = class_getInstanceMethod(c, sel);
  if (!m) return NULL;
  // ensure the override lands on the concrete class (not a superclass)
  if (!class_addMethod(c, sel, imp, method_getTypeEncoding(m))) return method_setImplementation(m, imp);
  return method_getImplementation(m);
}

__attribute__((destructor)) static void fini(void) { logf_("EXIT total_dispatches=%ld skipped=%ld", g_total_disp, g_skipped); }
__attribute__((constructor)) static void init(void) {
  @autoreleasepool {
    const char *d = getenv("MTL_DUMP_DIR"); g_dir = d ? @(d) : @"/tmp/mtl_dump";
    [[NSFileManager defaultManager] createDirectoryAtPath:g_dir withIntermediateDirectories:YES attributes:nil error:nil];
    g_log = fopen([[g_dir stringByAppendingString:@"/mtl_dump.log"] UTF8String], "a");
    const char *lim = getenv("MTL_DUMP_DISPATCH_LIMIT"); if (lim) g_dispatch_limit = atoi(lim);
    const char *sa = getenv("MTL_DUMP_SKIP_AFTER"); if (sa) g_skip_after = atol(sa);
    const char *sl = getenv("MTL_DUMP_SKIP_LIBS");
    if (sl) { char *dup = strdup(sl), *tok, *save = NULL; for (tok = strtok_r(dup, ",", &save); tok && g_nskip < 256; tok = strtok_r(NULL, ",", &save)) g_skip_libs[g_nskip++] = atoi(tok); free(dup); }
    const char *sh = getenv("MTL_DUMP_SKIP_HASHES");
    if (sh) { char *dup = strdup(sh), *tok, *save = NULL; for (tok = strtok_r(dup, ",", &save); tok && g_nskiph < 256; tok = strtok_r(NULL, ",", &save)) g_skip_hashes[g_nskiph++] = strtoull(tok, NULL, 16); free(dup); }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    Class dc = object_getClass(dev);
    orig_newLibSrc = (NewLibSrcIMP)swz(dc, @selector(newLibraryWithSource:options:error:), (IMP)my_newLibSrc);
    orig_newCPSFn = (NewCPSFnIMP)swz(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)my_newCPSFn);
    orig_newCPSDesc = (NewCPSDescIMP)swz(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)my_newCPSDesc);
    // encoder class: create a throwaway command buffer + encoder to learn the concrete class
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    Class ec = object_getClass(enc);
    orig_setPSO = (SetPSOIMP)swz(ec, @selector(setComputePipelineState:), (IMP)my_setPSO);
    orig_dispTG = (DispTGIMP)swz(ec, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)my_dispTG);
    orig_dispThreads = (DispTGIMP)swz(ec, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)my_dispThreads);
    [enc endEncoding];
    logf_("INIT skip_after=%ld nskip=%d nskiph=%d", g_skip_after, g_nskip, g_nskiph);
    logf_("INIT device=%s devclass=%s encclass=%s hooks lib=%d cpsfn=%d cpsdesc=%d setpso=%d disptg=%d dispth=%d",
          [[dev name] UTF8String], class_getName(dc), class_getName(ec), orig_newLibSrc != NULL, orig_newCPSFn != NULL,
          orig_newCPSDesc != NULL, orig_setPSO != NULL, orig_dispTG != NULL, orig_dispThreads != NULL);
  }
}
