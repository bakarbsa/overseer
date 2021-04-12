import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class Camera extends StatefulWidget {
  @override
  _CameraState createState() => _CameraState();
}

class _CameraState extends State<Camera> {
  bool isWorking = false;
  CameraController controller;
  List<CameraDescription> cameras;
  CameraImage imgCamera;
  String result = '';

  @override
  void initState() {
    super.initState();
    availableCameras().then((value) {
      cameras = value;
      if (cameras.length > 0) {
        controller = CameraController(cameras[1], ResolutionPreset.medium);
        controller.initialize().then((_) {
          if (!mounted) {
            return;
          }
          setState(() {
            controller.startImageStream((imageFromStream) {
              if (!isWorking) {
                isWorking = true;
                imgCamera = imageFromStream;
              }
            });
          });
        });
      } else {
        print('camera error');
      }
    }).catchError((e) {
      print(e.code);
    });
  }

  @override
  void dispose() {
    super.dispose();
    controller.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: cameraPreview(),
    );
  }

  Widget cameraPreview() {
    if (controller == null || !controller.value.isInitialized) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'loading',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 25.0,
                  fontWeight: FontWeight.w500),
            )
          ],
        ),
      );
    }
    return Stack(
        children: [
          RotationTransition(
              turns: AlwaysStoppedAnimation(0/360),
              child: CameraPreview(controller)
          ),
        ],
    );
  }
}