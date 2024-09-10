import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:hive/hive.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/widgets/classification_item.dart';
import 'package:flutter/material.dart';

class MyClassifications extends StatefulWidget {
  const MyClassifications({super.key});

  @override
  State<MyClassifications> createState() => _MyClassificationsState();
}

class _MyClassificationsState extends State<MyClassifications> {
  List<Map<dynamic, dynamic>> classificationHistory = [];
  late Stream<QuerySnapshot> historyStream;

  @override
  void initState() {
    super.initState();
    // Load data from Hive
    loadClassificationsFromHive();
    historyStream = FirebaseFirestore.instance
        .collection('classifications')
        .where('userId', isEqualTo: currentUser!.uid)
        .snapshots();
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
    });
  }

  DateTime convertTimestampToDateTime(int timestamp) {
    // Convert timestamp (in seconds) to milliseconds
    return DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          toolbarHeight: 70,
          automaticallyImplyLeading: false,
          title: const Padding(
            padding: EdgeInsets.all(8.0),
            child: const Text(
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
              Tab(text: 'On the Cloud'),
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
                              return ClassificationItem(
                                fromHive: true,
                                imageUrl: data['imageLocation'] ??
                                    'placeholder_image_url',
                                title:
                                    data['primaryClassification'] ?? 'No Title',
                                timestamp:
                                    data['timestamp'] ?? 'No Description',
                              );
                            },
                          ),
                  ),
                  Center(
                    child: FilledButton(
                      onPressed: () {},
                      child: const Text(
                        'Sync to Database',
                      ),
                    ),
                  )
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
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
                                child: Text('Error: ${snapshot.error}'));
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
                            return ClassificationItem(
                              fromHive: false,
                              imageUrl:
                                  data['imageUrl'] ?? 'placeholder_image_url',
                              title:
                                  data['primaryClassification'] ?? 'No Title',
                              timestamp: (data['timestamp']).toDate() ??
                                  'No Description',
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
    );
  }
}
