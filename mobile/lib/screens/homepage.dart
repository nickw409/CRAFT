import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/provider/login_provider.dart';
import 'package:craft/screens/about/about_tww.dart';
import 'package:craft/screens/edit_results.dart';
import 'package:craft/screens/my_classificatoins.dart';
import 'package:craft/screens/user_management/settings_page.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:hive/hive.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:page_transition/page_transition.dart';
import 'package:provider/provider.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:tflite_flutter_plus/tflite_flutter_plus.dart';

class HomePage extends StatefulWidget {
  static const String id = 'home';
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker picker = ImagePicker();
  File? selectedImage;
  String? classificaitonData;
  Position? currentPosition;
  img.Image? imageForModel;

  late Interpreter interpreter;
  late List<String> labels;

  late List<int> _inputShape;
  late List<int> _outputShape;

  late TensorImage _inputImage;
  late TensorBuffer _outputBuffer;

  late TfLiteType _inputType;
  late TfLiteType _outputType;

  final String _labelsFileName = 'assets/tww_labels.txt';

  Map<String, dynamic>? classificatoinMap;

  @override
  void initState() {
    super.initState();
    getCurrentUser();
    loadModel();
  }

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('ResNet152V2.tflite');

    _inputShape = interpreter.getInputTensor(0).shape;
    _outputShape = interpreter.getOutputTensor(0).shape;
    _inputType = interpreter.getInputTensor(0).type;
    _outputType = interpreter.getOutputTensor(0).type;
    _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);

    labels = await FileUtil.loadLabels(_labelsFileName);
  }

  TensorImage _preProcess() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.nearestneighbour))
        .add(NormalizeOp(0, 255))
        .build()
        .process(_inputImage);
  }

  Future<void> getCurrentUser() async {
    if (currentUser != null) {
      // Fetch user details from Firestore
      final userId = currentUser!.uid;
      final userDoc = await FirebaseFirestore.instance
          .collection('users')
          .doc(userId)
          .get();

      if (userDoc.exists) {
        final fullName = userDoc['name'] ?? 'User';
        final role = userDoc['role'] ?? 'Role';

        // Update the LoginProvider with the full name
        if (mounted) {
          final loginProvider =
              Provider.of<LoginProvider>(context, listen: false);
          loginProvider.login(fullName, role);
        }
      }
    }
  }

  Future pickAndCropImage(ImageSource source) async {
    final pickedImage =
        await picker.pickImage(source: source, imageQuality: 50);

    if (pickedImage == null) {
      return;
    }

    CroppedFile? croppedFile = await ImageCropper().cropImage(
      sourcePath: pickedImage.path,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: true),
        IOSUiSettings(
          aspectRatioLockEnabled: true,
          title: 'Crop Image',
        ),
      ],
    );

    if (croppedFile == null) {
      return;
    }

    //convert croppedFile to grayscale
    // Read the image file
    final File imageFile = File(croppedFile.path);
    final Uint8List imageBytes = await imageFile.readAsBytes();

    // Decode the image using the image package
    img.Image? image = img.decodeImage(imageBytes);

    if (image == null) {
      return;
    }

    // Convert the image to grayscale
    img.Image grayscaleImage = img.grayscale(image);
    imageForModel = grayscaleImage;

    // Encode the grayscale image back to PNG format
    final List<int> grayscaleBytes = img.encodePng(grayscaleImage);

    // Save the grayscale image as a new file
    final File grayscaleFile =
        await File(croppedFile.path).writeAsBytes(grayscaleBytes);

    setState(() {
      selectedImage = grayscaleFile;
    });
  }

  void resetScreen() {
    setState(() {
      selectedImage = null;
      classificaitonData = null;
      currentPosition = null;
    });
  }

  Position randomizePosition(Position position, double distanceMeters) {
    // Constants
    const double earthRadius = 6371000; // Earth's radius in meters

    // Convert distanceMeters meters to degrees
    double randomDistance = distanceMeters; // 500 meters
    double latOffset = (randomDistance / earthRadius) * (180 / pi);
    double lonOffset =
        (randomDistance / (earthRadius * cos(pi * position.latitude / 180))) *
            (180 / pi);

    // Generate random numbers to decide the direction of change
    double randomLat = (Random().nextDouble() * 2 - 1) * latOffset;
    double randomLon = (Random().nextDouble() * 2 - 1) * lonOffset;

    // Calculate new random position
    double newLatitude = position.latitude + randomLat;
    double newLongitude = position.longitude + randomLon;

    return Position(
      latitude: newLatitude,
      longitude: newLongitude,
      headingAccuracy: position.headingAccuracy,
      altitudeAccuracy: position.altitudeAccuracy,
      timestamp: position.timestamp,
      accuracy: position.accuracy,
      altitude: position.altitude,
      heading: position.heading,
      speed: position.speed,
      speedAccuracy: position.speedAccuracy,
    );
  }

  void classifyImage() async {
    Position pos = await _determinePosition();

    //randomize the position by 500 meters
    pos = randomizePosition(pos, 500);

    setState(() {
      currentPosition = pos;
      classificaitonData = "Classified";
    });

    _inputImage = TensorImage(_inputType);
    _inputImage.loadImage(imageForModel!);
    _inputImage = _preProcess();

    interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer());

    Map<String, double> resultMap = {};

    for (int i = 0; i < labels.length; i++) {
      resultMap[labels[i]] = _outputBuffer.getDoubleList()[i];
    }
    String highestConfidenceLabel = '';
    double highestConfidenceValue = 0.0;

    resultMap.forEach((label, value) {
      if (value > highestConfidenceValue) {
        highestConfidenceValue = value;
        highestConfidenceLabel = label;
      }
    });

    setState(() {
      classificatoinMap = {
        'primaryClassification': highestConfidenceLabel,
        'allClassificatoins': resultMap,
        'lattitude': pos.latitude,
        'longitude': pos.longitude,
      };
    });
  }

  Future<Position> _determinePosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Test if location services are enabled.
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      // Location services are not enabled don't continue
      // accessing the position and request users of the
      // App to enable the location services.
      return Future.error('Location services are disabled.');
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        // Permissions are denied, next time you could try
        // requesting permissions again (this is also where
        // Android's shouldShowRequestPermissionRationale
        // returned true. According to Android guidelines
        // your App should show an explanatory UI now.
        return Future.error('Location permissions are denied');
      }
    }

    if (permission == LocationPermission.deniedForever) {
      // Permissions are denied forever, handle appropriately.
      return Future.error(
          'Location permissions are permanently denied, we cannot request permissions.');
    }

    // When we reach here, permissions are granted and we can
    // continue accessing the position of the device.
    return await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.low,
        distanceFilter: 50,
      ),
    );
  }

  // Saving Data locally
  void saveClassificationLocally() async {
    // Open the Hive box for classifications
    var box = Hive.box('classificationBox');
    String imageLocation = selectedImage!.path;

    // Add additional fields to the classification map
    classificatoinMap!['timestamp'] = DateTime.now();
    classificatoinMap!['imageLocation'] = imageLocation;
    classificatoinMap!['latitude'] = currentPosition!.latitude;
    classificatoinMap!['longitude'] = currentPosition!.longitude;

    try {
      // Save the classification data locally to Hive
      await box.add(classificatoinMap!);
      resetScreen(); // Reset the screen after saving
    } catch (e) {
      _showError('Error saving classification: $e');
    }
  }

  void editClassification() async {
    Map<String, dynamic>? editedClassificatoin = await Navigator.push(
        context,
        PageTransition(
            child: EditResults(
              classificatoinMap: classificatoinMap,
            ),
            type: PageTransitionType.fade));

    setState(() {
      classificatoinMap = editedClassificatoin;
    });
  }

  void clearBox() {
    var box = Hive.box('classificationBox');

    box.clear();
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      child: Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: false,
          toolbarHeight: 80,
          title: const Padding(
            padding: EdgeInsets.all(8.0),
            child: Text(
              'CRAFT',
              style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 60,
                  fontWeight: FontWeight.w700),
            ),
          ),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              selectedImage == null
                  ? Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Expanded(
                              child: GestureDetector(
                                onTap: () =>
                                    pickAndCropImage(ImageSource.camera),
                                child: Container(
                                  height: MediaQuery.of(context).size.width / 3,
                                  decoration: BoxDecoration(
                                      color: Theme.of(context)
                                          .colorScheme
                                          .primary, // select color from current theme scheme
                                      borderRadius: const BorderRadius.all(
                                          Radius.circular(5))),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 40, vertical: 8),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          Icons.camera_alt_rounded,
                                          size: 90,
                                          color: Theme.of(context)
                                              .colorScheme
                                              .onPrimary,
                                        ),
                                        Text(
                                          'CAMERA',
                                          style: TextStyle(
                                            fontSize: 20,
                                            fontFamily: 'Uber',
                                            fontWeight: FontWeight.w900,
                                            color: Theme.of(context)
                                                .colorScheme
                                                .onPrimary,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: GestureDetector(
                                onTap: () =>
                                    pickAndCropImage(ImageSource.gallery),
                                child: Container(
                                  width:
                                      MediaQuery.of(context).size.width / 2.3,
                                  height: MediaQuery.of(context).size.width / 3,
                                  decoration: BoxDecoration(
                                      color: Theme.of(context)
                                          .colorScheme
                                          .primary, // select color from current theme scheme
                                      borderRadius: const BorderRadius.all(
                                          Radius.circular(5))),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 40, vertical: 8),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          Icons.image_rounded,
                                          size: 90,
                                          color: Theme.of(context)
                                              .colorScheme
                                              .onPrimary,
                                        ),
                                        Text(
                                          'GALLERY',
                                          style: TextStyle(
                                            fontSize: 20,
                                            fontFamily: 'Uber',
                                            fontWeight: FontWeight.w900,
                                            color: Theme.of(context)
                                                .colorScheme
                                                .onPrimary,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        GestureDetector(
                          onTap: () => Navigator.push(
                              context,
                              PageTransition(
                                  child: const MyClassifications(),
                                  type: PageTransitionType.fade)),
                          child: Container(
                            width: MediaQuery.of(context).size.width,
                            height: MediaQuery.of(context).size.width / 3,
                            decoration: BoxDecoration(
                                color: Theme.of(context)
                                    .colorScheme
                                    .primary, // select color from current theme scheme
                                borderRadius:
                                    const BorderRadius.all(Radius.circular(5))),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 40, vertical: 8),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.history_rounded,
                                    size: 90,
                                    color:
                                        Theme.of(context).colorScheme.onPrimary,
                                  ),
                                  Text(
                                    'CLASSIFICATION HISTORY',
                                    style: TextStyle(
                                      fontSize: 20,
                                      fontFamily: 'Uber',
                                      fontWeight: FontWeight.w900,
                                      color: Theme.of(context)
                                          .colorScheme
                                          .onPrimary,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: GestureDetector(
                                onTap: () {
                                  Navigator.push(
                                      context,
                                      PageTransition(
                                          child: const AboutTww(),
                                          type: PageTransitionType.fade));
                                },
                                child: Container(
                                  height: MediaQuery.of(context).size.width / 3,
                                  decoration: BoxDecoration(
                                      color: Theme.of(context)
                                          .colorScheme
                                          .primary, // select color from current theme scheme
                                      borderRadius: const BorderRadius.all(
                                          Radius.circular(5))),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 40, vertical: 8),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        FittedBox(
                                          fit: BoxFit.contain,
                                          child: Text(
                                            'ABOUT\nTUYSAYAN\nWHITE\nWARE',
                                            style: TextStyle(
                                              fontSize: 20,
                                              fontFamily: 'Uber',
                                              fontWeight: FontWeight.w900,
                                              color: Theme.of(context)
                                                  .colorScheme
                                                  .onPrimary,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: GestureDetector(
                                onTap: () {
                                  Navigator.push(
                                      context,
                                      PageTransition(
                                          child: SettingsPage(),
                                          type: PageTransitionType.fade));
                                },
                                child: Container(
                                  // width: MediaQuery.of(context).size.width / 2.3,
                                  height: MediaQuery.of(context).size.width / 3,
                                  decoration: BoxDecoration(
                                      color: Theme.of(context)
                                          .colorScheme
                                          .primary, // select color from current theme scheme
                                      borderRadius: const BorderRadius.all(
                                          Radius.circular(5))),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 40, vertical: 8),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          Icons.account_circle_rounded,
                                          size: 90,
                                          color: Theme.of(context)
                                              .colorScheme
                                              .onPrimary,
                                        ),
                                        Consumer<LoginProvider>(
                                          builder:
                                              (context, loginProvider, child) {
                                            if (loginProvider.isLoggedIn) {
                                              return Text(
                                                loginProvider.firstName!
                                                    .split(' ')[0]
                                                    .toUpperCase(),
                                                style: TextStyle(
                                                  fontSize: 20,
                                                  fontFamily: 'Uber',
                                                  fontWeight: FontWeight.w900,
                                                  color: Theme.of(context)
                                                      .colorScheme
                                                      .onPrimary,
                                                ),
                                              );
                                            } else {
                                              return Text(
                                                "GUEST",
                                                style: TextStyle(
                                                  fontSize: 20,
                                                  fontFamily: 'Uber',
                                                  fontWeight: FontWeight.w900,
                                                  color: Theme.of(context)
                                                      .colorScheme
                                                      .onPrimary,
                                                ),
                                              );
                                            }
                                          },
                                        ),
                                        // Text(
                                        //   'SETTINGS',
                                        //   style: TextStyle(
                                        //     fontSize: 20,
                                        //     fontFamily: 'Uber',
                                        //     fontWeight: FontWeight.w900,
                                        //     color: Theme.of(context)
                                        //         .colorScheme
                                        //         .onPrimary,
                                        //   ),
                                        // ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    )
                  : Container(),
              selectedImage != null
                  ? AspectRatio(
                      aspectRatio: 1,
                      child: Container(
                        decoration: BoxDecoration(
                            color: Theme.of(context)
                                .colorScheme
                                .secondaryContainer, // select color from current theme scheme
                            borderRadius:
                                const BorderRadius.all(Radius.circular(5))),
                        width: MediaQuery.of(context).size.width,
                        // height: 100,
                        child: Center(
                            child: selectedImage != null
                                ? ClipRRect(
                                    borderRadius: BorderRadius.circular(5),
                                    child: Image.file(
                                      selectedImage!,
                                      fit: BoxFit.cover,
                                      width: MediaQuery.of(context).size.width,
                                    ))
                                : Container()),
                      ),
                    )
                  : Container(),
              const SizedBox(
                height: 30,
              ),
              classificaitonData != null
                  ? Container(
                      decoration: BoxDecoration(
                          color: Theme.of(context)
                              .colorScheme
                              .primaryContainer, // select color from current theme scheme
                          borderRadius:
                              const BorderRadius.all(Radius.circular(5))),
                      width: MediaQuery.of(context).size.width,
                      child: Padding(
                          padding: const EdgeInsets.all(12.0),
                          child: Column(
                            children: [
                              const Text('Primary Classification:'),
                              Text(
                                "${classificatoinMap!['primaryClassification'].toString()} [${classificatoinMap!['allClassificatoins']?[classificatoinMap!['primaryClassification'].toString()].toStringAsFixed(3)}]",
                                style: const TextStyle(
                                    fontSize: 20, fontWeight: FontWeight.bold),
                              ),
                              Text(
                                  "Location: ${classificatoinMap!['lattitude'].toStringAsFixed(4)}, ${classificatoinMap!['longitude'].toStringAsFixed(4)}"),
                              // const Text("Other Classifications:"),
                            ],
                          )),
                    )
                  : Container(),
              const SizedBox(
                height: 16,
              ),
              selectedImage != null && classificaitonData == null
                  ? Center(
                      child: FilledButton(
                          onPressed: classifyImage,
                          child: const Text('Classify')),
                    )
                  : Container(),
              selectedImage != null && classificaitonData == null
                  ? Center(
                      child: TextButton(
                          onPressed: resetScreen,
                          child: const Text('Clear Image')),
                    )
                  : Container(),
              selectedImage != null && classificaitonData != null
                  ? Center(
                      child: FilledButton(
                          onPressed: saveClassificationLocally,
                          child: const Text('Save Classification')),
                    )
                  : Container(),
              selectedImage != null && classificaitonData != null
                  ? Center(
                      child: TextButton(
                          onPressed: resetScreen,
                          child: const Text('Clear Image and Classification')),
                    )
                  : Container(),
              selectedImage != null && classificaitonData != null
                  ? Center(
                      child: TextButton(
                          onPressed: editClassification,
                          child: const Text('Edit Classification')),
                    )
                  : Container(),

              // for testing purposes
              // Center(
              //   child: FilledButton(
              //       onPressed: () {
              //         var box = Hive.box('classificationBox');

              //         box.clear();
              //       },
              //       child: const Text('Clear Local Storage')),
              // ),
            ],
          ),
        ),
      ),
    );
  }
}
