import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/provider/login_provider.dart';
import 'package:craft/screens/user_management/login_page.dart';
import 'package:craft/widgets/sherd_details.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:hive/hive.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/widgets/classification_item.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';
import 'package:provider/provider.dart';

class MyClassifications extends StatefulWidget {
  const MyClassifications({super.key});

  @override
  State<MyClassifications> createState() => _MyClassificationsState();
}

class _MyClassificationsState extends State<MyClassifications> {
  List<Map<dynamic, dynamic>> classificationHistory = [];
  late Stream<QuerySnapshot> historyStream;
  late int numClassifications;
  late bool isLoggedIn;

  @override
  void initState() {
    super.initState();
    // Load data from Hive
    loadClassificationsFromHive();

    final userProvider = Provider.of<LoginProvider>(context, listen: false);

    isLoggedIn = userProvider.isLoggedIn;

    if (isLoggedIn) {
      // Load data from Firestore
      historyStream = FirebaseFirestore.instance
          .collection('classifications')
          .where('userId', isEqualTo: currentUser?.uid)
          .snapshots();
    }
  }

  void loadClassificationsFromHive() async {
    var box = Hive.box('classificationBox');

    // Fetch all classifications
    List<Map<dynamic, dynamic>> classifications = [];
    for (var i = 0; i < box.length; i++) {
      classifications.add(box.getAt(i));
    }

    setState(() {
      classificationHistory = classifications;
      numClassifications = classifications.length;
    });
  }

  DateTime convertTimestampToDateTime(int timestamp) {
    // Convert timestamp (in seconds) to milliseconds
    return DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
  }

  Future<void> syncToDatabase() async {
    // Show loading spinner
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return const Center(
          child: CircularProgressIndicator(),
        );
      },
    );

    var box = Hive.box('classificationBox');
    final FirebaseStorage storage = FirebaseStorage.instance;

    // Fetch all classifications
    List<Map<dynamic, dynamic>> classifications = [];
    for (var i = 0; i < box.length; i++) {
      classifications.add(box.getAt(i));
    }

    // Add classifications to Firestore
    for (var classification in classifications) {
      File imageFile = File(classification['imageLocation']);
      String fileName = imageFile.uri.pathSegments.last;
      Reference storageRef = storage.ref().child('images/$fileName');

      UploadTask uploadTask = storageRef.putFile(imageFile);
      TaskSnapshot taskSnapshot = await uploadTask;

      // Get the download URL of the uploaded image
      String imageUrl = await taskSnapshot.ref.getDownloadURL();

      await FirebaseFirestore.instance.collection('classifications').add({
        'userId': currentUser!.uid,
        'primaryClassification': classification['primaryClassification'],
        'allClassificatoins': classification['allClassificatoins'],
        'timestamp': classification['timestamp'],
        'imageUrl': imageUrl,
        'latitude': classification['latitude'],
        'longitude': classification['longitude'],
      });
    }

    // Clear Hive box
    await box.clear();

    // Reload data from Hive
    loadClassificationsFromHive();

    if (mounted) {
      Navigator.of(context).pop();
    }

    _showMessage('Synced to database successfully.');
  }

  void _showMessage(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Success'),
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
    return DefaultTabController(
      length: 2,
      child: SafeArea(
        child: Scaffold(
          appBar: AppBar(
            toolbarHeight: 80,
            title: const Padding(
              padding: EdgeInsets.all(8.0),
              child: Text(
                'History',
                style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 60,
                    fontWeight: FontWeight.w700),
              ),
            ),
            bottom: const TabBar(
              tabs: [
                Tab(text: 'On My Device'),
                Tab(text: 'On The Cloud'),
              ],
            ),
          ),
          body: TabBarView(
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Expanded(
                      child: classificationHistory.isEmpty
                          ? const Center(child: Text('No history found.'))
                          : ListView.builder(
                              itemCount: classificationHistory.length,
                              itemBuilder: (context, index) {
                                var data = classificationHistory[index];
                                return GestureDetector(
                                  onTap: () {
                                    showSherdDetailsDialog(
                                      context,
                                      imageUrl: data['imageLocation'],
                                      title: data['primaryClassification'],
                                      details: data['allClassificatoins'],
                                      timestamp: data['timestamp'],
                                      fromHive: true,
                                      lattitude: data['latitude'],
                                      longitude: data['longitude'],
                                    );
                                  },
                                  child: ClassificationItem(
                                    fromHive: true,
                                    imageUrl: data['imageLocation'],
                                    title: data['primaryClassification'],
                                    timestamp: data['timestamp'],
                                  ),
                                );
                              },
                            ),
                    ),
                    Consumer<LoginProvider>(
                      builder: (context, loginProvider, child) {
                        if (loginProvider.isLoggedIn) {
                          return (numClassifications > 0)
                              ? Center(
                                  child: FilledButton(
                                    onPressed: syncToDatabase,
                                    child: const Text(
                                      'Sync to Database',
                                    ),
                                  ),
                                )
                              : Container();
                        } else {
                          return Center(
                            child: FilledButton(
                              onPressed: () => Navigator.push(
                                  context,
                                  PageTransition(
                                      child: const LoginPage(),
                                      type: PageTransitionType.fade)),
                              child: const Text('Sign in to Save to the Cloud'),
                            ),
                          );
                        }
                      },
                    ),
                  ],
                ),
              ),

              Padding(
                padding: const EdgeInsets.all(8.0),
                child: (!isLoggedIn)
                    ? const Center(
                        child: Text('Please sign in to view cloud history.'),
                      )
                    : Column(
                        children: [
                          Expanded(
                            child: StreamBuilder(
                                stream: historyStream,
                                builder: (context, snapshot) {
                                  if (snapshot.connectionState ==
                                      ConnectionState.waiting) {
                                    return const Center(
                                        child: CircularProgressIndicator());
                                  } else if (snapshot.hasError) {
                                    return Center(
                                        child:
                                            Text('Error: ${snapshot.error}'));
                                  } else if (!snapshot.hasData ||
                                      snapshot.data!.docs.isEmpty) {
                                    return const Center(
                                        child: Text('No history found.'));
                                  }

                                  List<QueryDocumentSnapshot> documents =
                                      snapshot.data!.docs;

                                  // Build the list of ClassificationItem widgets
                                  List<Widget> items = documents.map((doc) {
                                    Map<String, dynamic> data =
                                        doc.data() as Map<String, dynamic>;
                                    return GestureDetector(
                                      onTap: () {
                                        showSherdDetailsDialog(
                                          context,
                                          imageUrl: data['imageUrl'],
                                          title: data['primaryClassification'],
                                          details: data['allClassificatoins'],
                                          timestamp:
                                              (data['timestamp']).toDate(),
                                          fromHive: false,
                                          lattitude: data['latitude'],
                                          longitude: data['longitude'],
                                        );
                                      },
                                      child: ClassificationItem(
                                        fromHive: false,
                                        imageUrl: data['imageUrl'],
                                        title: data['primaryClassification'],
                                        timestamp: (data['timestamp']).toDate(),
                                      ),
                                    );
                                  }).toList();

                                  return ListView(
                                    children: items,
                                  );
                                }),
                          ),
                        ],
                      ),
              ),
              // Tab 1: Hive History
            ],
          ),
        ),
      ),
    );
  }
}

/*
 showSherdDetailsDialog(
                                        context,
                                        imageUrl: data['imageUrl'],
                                        title: data['primaryClassification'],
                                        details: data['allClassificatoins'],
                                        timestamp: (data['timestamp']).toDate(),
                                        fromHive: true,
                                      );
*/
