import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/widgets/classification_item.dart';
import 'package:flutter/material.dart';

class MyClassificatoins extends StatefulWidget {
  const MyClassificatoins({super.key});

  @override
  State<MyClassificatoins> createState() => _MyClassificatoinsState();
}

class _MyClassificatoinsState extends State<MyClassificatoins> {
  late Stream<QuerySnapshot> historyStream;

  @override
  void initState() {
    super.initState();
    // Initialize the stream
    historyStream = FirebaseFirestore.instance
        .collection('classifications')
        .where('userId', isEqualTo: currentUser!.uid)
        .snapshots();
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
              height: 16,
            ),
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: FittedBox(
                fit: BoxFit.contain,
                child: Text(
                  'History',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 60,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            // const SizedBox(
            //   height: 16,
            // ),
            // const Divider(),s
            Expanded(
              child: StreamBuilder(
                  stream: historyStream,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    } else if (snapshot.hasError) {
                      return Center(child: Text('Error: ${snapshot.error}'));
                    } else if (!snapshot.hasData ||
                        snapshot.data!.docs.isEmpty) {
                      return const Center(child: Text('No history found.'));
                    }

                    List<QueryDocumentSnapshot> documents = snapshot.data!.docs;

                    // Build the list of ClassificationItem widgets
                    List<Widget> items = documents.map((doc) {
                      Map<String, dynamic> data =
                          doc.data() as Map<String, dynamic>;
                      return ClassificationItem(
                        imageUrl: data['imageUrl'] ?? 'placeholder_image_url',
                        title: data['primaryClassification'] ?? 'No Title',
                        timestamp: data['timestamp'] ?? 'No Description',
                      );
                    }).toList();

                    return ListView(
                      children: items,
                    );
                  }),
            )
          ],
        ),
      ),
    );
  }
}
