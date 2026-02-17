// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "face_detection_tflite",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "face-detection-tflite", targets: ["face_detection_tflite"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "face_detection_tflite",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
