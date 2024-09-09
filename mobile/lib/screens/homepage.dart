import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/provider/login_provider.dart';
import 'package:craft/screens/about/about_tww.dart';
import 'package:craft/screens/edit_results.dart';
import 'package:craft/screens/my_classificatoins.dart';
import 'package:craft/screens/user_management/settings_page.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:page_transition/page_transition.dart';
import 'package:provider/provider.dart';

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

  Map<String, dynamic>? classificatoinMap;

  @override
  void initState() {
    super.initState();
    getCurrentUser();
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
        final fullName = userDoc['name'] ??
            'User'; // Assuming fullName is a field in Firestore

        // Update the LoginProvider with the full name
        if (mounted) {
          final loginProvider =
              Provider.of<LoginProvider>(context, listen: false);
          loginProvider.login(fullName);
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

    setState(() {
      selectedImage = File(croppedFile.path);
    });
  }

  void resetScreen() {
    setState(() {
      selectedImage = null;
      classificaitonData = null;
      currentPosition = null;
    });
  }

  void classifyImage() async {
    Position pos = await _determinePosition();

    classificatoinMap = {
      'primaryClassification': 'Flagstaff',
      'allClassificatoins': {
        'Kana\'a': 0.23,
        'Black Mesa': 0.01,
        'Sosi': 0.20,
        'Dogoszhi': 1.2,
        'Flagstaff': 0.65,
        'Tuysayan': 2.3,
        'Kayenta': 4.5,
      },
      'lattitude': pos.latitude,
      'longitude': pos.longitude,
    };

    setState(() {
      currentPosition = pos;
      classificaitonData =
          "Flagstaff: Confidence 0.123\nBlack Mesa: Confidence 0.123\nKnaa: Confidence 0.123";
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
        // using the low locaiton accuracy makes accuracy of location 0-1000 m
        locationSettings:
            const LocationSettings(accuracy: LocationAccuracy.low));
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(
              height: 32,
            ),
            const FittedBox(
              fit: BoxFit.contain,
              child: Text(
                'CRAFT',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 60,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            // Check if user is logged in and display a welcome message
            Consumer<LoginProvider>(
              builder: (context, loginProvider, child) {
                if (loginProvider.isLoggedIn) {
                  return Text(
                    'Welcome, ${loginProvider.firstName}!',
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  );
                } else {
                  return const Text(
                    'You are not logged in!',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  );
                }
              },
            ),
            const SizedBox(
              height: 16,
            ),
            selectedImage == null
                ? Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          Expanded(
                            child: GestureDetector(
                              onTap: () => pickAndCropImage(ImageSource.camera),
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
                                    mainAxisAlignment: MainAxisAlignment.center,
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
                                width: MediaQuery.of(context).size.width / 2.3,
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
                                    mainAxisAlignment: MainAxisAlignment.center,
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
                                child: const MyClassificatoins(),
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
                                    color:
                                        Theme.of(context).colorScheme.onPrimary,
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
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Text(
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
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(
                                        Icons.settings_rounded,
                                        size: 90,
                                        color: Theme.of(context)
                                            .colorScheme
                                            .onPrimary,
                                      ),
                                      Text(
                                        'SETTINGS',
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
                            Text(
                                "Primary Classification: ${classificatoinMap!['primaryClassification'].toString()} [${classificatoinMap!['allClassificatoins']?[classificatoinMap!['primaryClassification'].toString()].toString()}]"),
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
                ? FilledButton(
                    onPressed: classifyImage, child: const Text('Classify'))
                : Container(),
            selectedImage != null && classificaitonData == null
                ? TextButton(
                    onPressed: resetScreen, child: const Text('Clear Image'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? FilledButton(
                    onPressed: classifyImage,
                    child: const Text('Save Classification'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? TextButton(
                    onPressed: resetScreen,
                    child: const Text('Clear Image and Classification'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? TextButton(
                    onPressed: editClassification,
                    child: const Text('Edit Classification'))
                : Container(),
          ],
        ),
      ),
    );
  }
}
