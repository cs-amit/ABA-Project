# Galaxy Watch4 local-development setup

Use these steps only for the project owner's test watch.

1. On the watch, enable developer options and turn on ADB debugging (and wireless debugging when pairing over Wi-Fi).
2. Pair the watch with Android Studio or ADB, then confirm it appears as an authorized device before deploying the Wear app.
3. Update Samsung Health Sensor Service to the version required by Samsung Health Sensor SDK 1.4.1.
4. In Samsung Health Sensor Service, enable developer mode on this owner-controlled test device before testing sensor collection.

> Warning: Samsung Health Sensor Service developer mode is for the owner's test device only. Public distribution requires Samsung partner registration and app-signature approval.

The prototype supports Galaxy Watch4 and later Wear OS powered by Samsung watches only. If required sensor capabilities are unavailable, the app must show them as unsupported rather than guessing their availability.
