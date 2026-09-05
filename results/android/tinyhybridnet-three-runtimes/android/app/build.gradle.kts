plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.b3bench"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.b3bench"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    androidResources { noCompress += listOf("pte", "tflite", "onnx", "bin") }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = false } }
}

dependencies {
    implementation("org.pytorch:executorch-android:1.4.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("junit:junit:4.13.2")
}
