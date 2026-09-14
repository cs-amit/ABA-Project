import com.android.build.api.dsl.ApplicationExtension
import java.security.KeyStore
import java.security.cert.X509Certificate

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.aba.smartsleep.wear"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.aba.smartsleep"
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

@Suppress("UNCHECKED_CAST")
val appDebugApplicationId = rootProject.extensions.extraProperties["appDebugApplicationId"]
    as org.gradle.api.provider.Property<String>

fun debugSigningCertificate(module: Project): X509Certificate {
    val android = module.extensions.getByType(ApplicationExtension::class.java)
    val signing = android.signingConfigs.getByName("debug")
    val storeFile = signing.storeFile
        ?: File(System.getProperty("user.home"), ".android/debug.keystore")
    val storePassword = signing.storePassword ?: "android"
    val keyAlias = signing.keyAlias ?: "androiddebugkey"

    check(storeFile.isFile) {
        "${module.path} debug keystore is missing: ${storeFile.absolutePath}"
    }

    val keyStore = KeyStore.getInstance(signing.storeType ?: KeyStore.getDefaultType())
    storeFile.inputStream().use { keyStore.load(it, storePassword.toCharArray()) }
    return keyStore.getCertificate(keyAlias) as? X509Certificate
        ?: error("${module.path} debug signing certificate is missing for alias '$keyAlias'.")
}

val verifyWearDataLayerPackageIdentity by tasks.registering {
    doLast {
        val appApplicationId = appDebugApplicationId.orNull
            ?: error(":app did not expose a resolved debug application ID.")
        val wearApplicationId = android.defaultConfig.applicationId
            ?: error(":wear debug application ID is not configured.")
        check(wearApplicationId == appApplicationId) {
            "Wear debug application ID must match :app's resolved debug application ID for Wear Data Layer: " +
                "expected $appApplicationId but was $wearApplicationId."
        }

        val appCertificate = debugSigningCertificate(project(":app"))
        val wearCertificate = debugSigningCertificate(project)
        check(appCertificate.encoded.contentEquals(wearCertificate.encoded)) {
            "Wear debug signing certificate must match :app's debug signing certificate for Wear Data Layer."
        }
    }
}

tasks.named("preBuild") {
    dependsOn(verifySamsungHealthSensorAar)
    dependsOn(verifyWearDataLayerPackageIdentity)
}

dependencies {
    implementation(project(":core"))
    implementation(libs.androidx.wear.compose.material3)
    implementation(libs.kotlinx.coroutines.android)
    implementation(files("libs/samsung-health-sensor-api.aar"))
    implementation("com.google.android.gms:play-services-wearable:20.0.1")
    testImplementation(libs.junit4)
}
