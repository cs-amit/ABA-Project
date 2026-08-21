plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.aba.smartsleep.wear"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.aba.smartsleep.wear"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildFeatures { compose = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { jvmToolchain(17) }

val samsungHealthSensorAar = file("libs/samsung-health-sensor-api.aar")
val verifySamsungHealthSensorAar by tasks.registering {
    doLast {
        check(samsungHealthSensorAar.isFile) {
            "Required Samsung Health Sensor SDK 1.4.1 AAR is missing: ${samsungHealthSensorAar.absolutePath}"
        }
    }
}

tasks.named("preBuild") {
    dependsOn(verifySamsungHealthSensorAar)
}

dependencies {
    implementation(project(":core"))
    implementation(libs.androidx.wear.compose.material3)
    implementation(libs.kotlinx.coroutines.android)
    implementation(files("libs/samsung-health-sensor-api.aar"))
    implementation("com.google.android.gms:play-services-wearable:20.0.1")
}
