import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class ClassificationItem extends StatelessWidget {
  const ClassificationItem(
      {super.key,
      required this.imageUrl,
      required this.title,
      required this.timestamp});

  final String imageUrl;
  final String title;
  final Timestamp timestamp;

  String formatTimestamp(Timestamp timestamp) {
    // Convert Timestamp to DateTime
    DateTime dateTime = timestamp.toDate();

    // Create a DateFormat object for MM/DD/YY
    DateFormat formatter = DateFormat('MM/dd/yy');

    // Format the DateTime to a string
    return formatter.format(dateTime);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(5),
              child: Image.network(
                'https://via.placeholder.com/150',
                width: 75,
              ),
            ),
            const SizedBox(
              width: 16,
            ),
            Column(
              mainAxisAlignment: MainAxisAlignment.start,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 30,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                Text(
                  formatTimestamp(timestamp),
                  style: const TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 20,
                  ),
                ),
              ],
            ),
          ],
        ),
        const Divider(),
      ],
    );
  }
}
