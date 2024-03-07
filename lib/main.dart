import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/scheduler.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image/image.dart' as img;
//Info menu items import
import 'tww_info.dart';
import 'kanaa.dart';
import 'black_mesa.dart';
import 'sosi.dart';
import 'dogoszhi.dart';
import 'flagstaff.dart';
import 'tusayan.dart';
import 'kayenta.dart';
import 'about.dart';


void main() {

  //Set expiration date if appropriate
  var expirationDate = DateFormat('yyyy-MM-dd').parse('2024-03-31');

  var now = DateTime.now();
  if (now.isAfter(expirationDate)) {
  //if (1==0) { //Commenting the above line, and uncommenting this, results in an app with no expiration
      runApp(MaterialApp(home:ExpiredPage()));
    } else {

      runApp(MyApp());}
  }





class MyApp extends StatelessWidget {


  MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'TWW Classifier',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.red),
          useMaterial3: true,
        ),
        debugShowCheckedModeBanner: false,
        home: Home());
  }
}

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  bool _loading = true;
  File? _image; //path for displayed/classified image
  List? _output; //Direct classification output from model; adapted into result variable
  String result = 'Classification result will go here.'; //Shows top 3 results above threshold
  String sysdate = "";
  String systime = "";


  final picker = ImagePicker();

  @override
  void initState() {
        //Initializes app, including loading CNN model
    super.initState();

     loadModel().then((value) {
      setState(() {});
    });
  }

//Routine that takes loaded image, formats it, then classifies it
  classifyImage(File image) async {
    final bytes = await image.readAsBytes();
    img.Image? original = img.decodeImage(bytes);


    // Calculate the center of the image
    int centerX = original!.width ~/ 2;
    int centerY = original.height ~/ 2;

    // Calculate the side length of the square
    int sideLength = original.width < original.height
        ? original.width
        : original.height;

    // Calculate the top-left coordinates of the square
    int topLeftX = centerX - (sideLength ~/ 2);
    int topLeftY = centerY - (sideLength ~/ 2);

    // Crop the image into square
    img.Image croppedImage = img.copyCrop(original, x: topLeftX,
        y: topLeftY,
        height: sideLength,
        width: sideLength);

    img.Image gray_image = img.grayscale(croppedImage); //Converts color image to grayscale to eliminate color effects
    img.Image resized = img.copyResize(gray_image, width: 224, height: 224); //Resizes image to max required by model

    final encoded_bytes = img.encodeJpg(resized);//Encodes image into jpg
    final path = '${_image
        ?.path}'; //Sets the save path for the image to the location in cache
    File(path).writeAsBytesSync(encoded_bytes); //Writes cropped grayscale image into cache for display
    var duration = const Duration(
        milliseconds: 120); //Pause to allow save to complete
    sleep(duration);

    double meanSubtract = 127.5; //ResNet152V2 requires image originally in 0 to 255 to be -1.0 to 1.0
    double stdDivide = 127.5; //Subtracting 127.5 from each pixel, then dividing by 127.5, sets image into proper format

    var output = await Tflite
        .runModelOnImage( //Classification. Mean and threshold set to keep image values intact.
      path: path,
      // required
      numResults: 3,
      threshold: 0.1,
      imageMean: meanSubtract,
      imageStd: stdDivide,
    );


    result = ""; //Loads top 3 classification info into result
    int j = output!.length;
    result = '${output[0]['label']}  Confidence: ${output[0]['confidence']
        .toStringAsFixed(3)} \n';
    if (j > 1) {
      result += '${output[1]['label']}  Confidence: ${output[1]['confidence']
          .toStringAsFixed(3)} \n';
    }
    if (j > 2) {
      result += '${output[2]['label']}  Confidence: ${output[2]['confidence']
          .toStringAsFixed(3)} \n';
    }
    //result += _image!.path;


    setState(() { //Resets state to display image, modify results
      _output = output;
      result;
      _loading = false;
    });
  }

//Loads in initial data (CNN model, date
  loadModel() async {
    //Loads CNN model and labels
    await Tflite.loadModel(
        model: 'assets/ResNet152V2.tflite', labels: 'assets/tww_labels.txt'); //tflite is the CNN model, tww_labels.txt lists TWW types in order
  }

  get_DateTime() async {
    DateTime now = DateTime.now();
    sysdate = DateFormat('yyyy-MM-dd – kk:mm').format(now);
    systime = DateFormat('kk-mm').format(now);
  }

  @override
  void dispose() {
    super.dispose();
  }

  //Takes picture with camera
  pickImage() async {
    //Takes picture with camera
    var image = await picker.pickImage(source: ImageSource.camera);
    if (image == null) return null;


    setState(() {
      _output = null;
      _image = File(image.path);
    });

    classifyImage(_image!); //Process and classify image
  }
//Selects picture on phone using gallery app
  pickGalleryImage() async {
    //Selects image from gallery
    var image = await picker.pickImage(source: ImageSource.gallery);
    if (image == null) return null;


    setState(() {
      _output = null;
      _image = File(image.path);
    });

    classifyImage(_image!); //Process and classify image
  }

  @override
  //Following builds up app interface, including top bar, info menu, and buttons
  Widget build(BuildContext context) {

    return Scaffold(
        backgroundColor: Colors.grey[300],
        appBar: AppBar(
          title: Text(
              'Tusayan WW Typer Resnet152V2', style: TextStyle(fontSize: 18.0)),
          centerTitle: true,
          backgroundColor: Colors.blue,
          elevation: 0.0,
          actions: [
//Creates info menu
            PopupMenuButton(
              // add icon, by default "3 dot" icon
              // icon: Icon(Icons.book)
                itemBuilder: (context) {
                  return [
                    PopupMenuItem<int>(
                      value: 0,
                      child: Text("TWW info"),
                    ),

                    PopupMenuItem<int>(
                      value: 1,
                      child: Text("Kana'a"),
                    ),

                    PopupMenuItem<int>(
                      value: 2,
                      child: Text("Black Mesa"),
                    ),
                    PopupMenuItem<int>(
                      value: 3,
                      child: Text("Sosi"),
                    ),
                    PopupMenuItem<int>(
                      value: 4,
                      child: Text("Dogoszhi"),
                    ),
                    PopupMenuItem<int>(
                      value: 5,
                      child: Text("Flagstaff"),
                    ),
                    PopupMenuItem<int>(
                      value: 6,
                      child: Text("Tusayan"),
                    ),
                    PopupMenuItem<int>(
                      value: 7,
                      child: Text("Kayenta"),
                    ),
                    PopupMenuItem<int>(
                      value: 8,
                      child: Text("About"),
                    ),
                  ];
                },
                onSelected: (value) {
                  if (value == 0) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => tww_info()));
                  } else if (value == 1) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => kanaa()));
                  } else if (value == 2) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => black_mesa()));
                  } else if (value == 3) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => sosi()));
                  } else if (value == 4) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => dogoszhi()));
                  } else if (value == 5) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => flagstaff()));
                  } else if (value == 6) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => tusayan()));
                  } else if (value == 7) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => kayenta()));
                  } else if (value == 8) {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => about()));
                  }
                }
            ),


          ],
        ),
        body: Container(
            padding: EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                SizedBox(height: 50),
                Padding(
                  padding: const EdgeInsets.all(50),
                  child: Column(
                    children: [
                      //Main title section - will make this the site ID
                      Text(
                        '  ', //Text above image box
                        style: TextStyle(
                            color: Colors.brown,
                            fontWeight: FontWeight.w500,
                            fontSize: 20),
                      ),

                    ],
                  ),
                ),
                Center(
                  child: _loading
                      ? Container(
                    width: 350,
                    child: Column(
                      children: <Widget>[ //Displays image
                        Padding(
                          padding: const EdgeInsets.all(50),
                          child: Container(
                              decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(10),
                                  color: Colors.white,
                                  boxShadow: [
                                    BoxShadow(
                                      offset: Offset(4, 4),
                                      blurRadius: 15,
                                      spreadRadius: 1,
                                      color: Colors.grey.shade500,
                                    ),
                                    BoxShadow(
                                        offset: Offset(-4, -4),
                                        blurRadius: 15,
                                        spreadRadius: 1,
                                        color: Colors.white),
                                  ]),
                              child: Padding(
                                padding: const EdgeInsets.all(20),
                                child: const Icon(Icons.image,
                                    size: 100),
                              )),
                        ),
                        SizedBox(height: 50),
                      ],
                    ),
                  )
                      : Container( //Displays results
                    child: Column(
                      children: <Widget>[
                        Container(
                          height: 250,
                          child: Image.file(_image!),
                        ),
                        SizedBox(height: 20),
                        _output != null
                            ?


                        Text(

                          result,
                          style: TextStyle(
                            fontWeight: FontWeight.w500,
                            fontSize: 16,
                          ),
                        )
                            : Container(),
                        SizedBox(height: 10),
                      ],
                    ),
                  ),
                ),
                Container(
                  width: MediaQuery
                      .of(context)
                      .size
                      .width,
                  child: Row( //Displays Camera and Gallery button
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: <Widget>[ //Camera button
                      GestureDetector( //Camera button
                        onTap: () {
                          pickImage();
                        },
                        child: Container(
                            width: MediaQuery
                                .of(context)
                                .size
                                .width / 3,
                            alignment: Alignment.center,
                            padding: EdgeInsets.symmetric(
                                horizontal: 10, vertical: 18),
                            decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(10),
                                boxShadow: [
                                  BoxShadow(
                                    offset: Offset(4, 4),
                                    blurRadius: 15,
                                    spreadRadius: 1,
                                    color: Colors.grey.shade500,
                                  ),
                                  BoxShadow(
                                      offset: Offset(-4, -4),
                                      blurRadius: 15,
                                      spreadRadius: 1,
                                      color: Colors.white),
                                ]),
                            child: Column(
                              children: [
                                Icon(
                                  Icons.photo_camera,
                                  size: 30,
                                  color: Colors.brown,
                                ),
                                Text("Camera")
                              ],
                            )),
                      ),
                      SizedBox(width: 5),
                      GestureDetector(
                        onTap: () {
                          pickGalleryImage();
                        },
                        child: Container( //Gallery button
                            width: MediaQuery
                                .of(context)
                                .size
                                .width / 3,
                            alignment: Alignment.center,
                            padding: EdgeInsets.symmetric(
                                horizontal: 10, vertical: 18),
                            decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(10),
                                boxShadow: [
                                  BoxShadow(
                                    offset: Offset(4, 4),
                                    blurRadius: 15,
                                    spreadRadius: 1,
                                    color: Colors.grey.shade500,
                                  ),
                                  BoxShadow(
                                      offset: Offset(-4, -4),
                                      blurRadius: 15,
                                      spreadRadius: 1,
                                      color: Colors.white),
                                ]),
                            child: Column(
                              children: [
                                Icon(
                                  Icons.photo,
                                  size: 30,
                                  color: Colors.brown,
                                ),
                                Text("Gallery")
                              ],
                            )),
                      ),
                    ],
                  ),
                ),
              ],
            )));
  }
}

void closeAppUsingExit() {
  exit(0);
}

//This page displays if the expiration date has passed
class ExpiredPage extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Expired',
              style: TextStyle(
                fontSize: 32,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 16),
            Text(
              'This app has expired on ${_formatExpiryDate(DateTime(2023, 1, 1))}. Please contact leszek.pawlowicz@nau.edu for latest version.',
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 32),
            ElevatedButton(
              onPressed: () {closeAppUsingExit();
            // Close app
              },
              child: Text('Close'),
            ),
          ],
        ),
      ),
    );
  }

  String _formatExpiryDate(DateTime date) => DateFormat('yyyy-MM-dd').format(date);

}