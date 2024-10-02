import 'dart:io';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class ClassificationItem extends StatelessWidget {
  const ClassificationItem({
    super.key,
    required this.imageUrl,
    required this.title,
    required this.timestamp,
    required this.fromHive,
  });

  final String imageUrl;
  final String title;
  final DateTime timestamp;
  final bool fromHive;

  String formatTimestamp(DateTime timestamp) {
    // Convert Timestamp to DateTime
    DateTime dateTime = timestamp;

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
              child: fromHive
                  ? Image.file(
                      File(imageUrl),
                      width: 75,
                    )
                  : Image.network(
                      imageUrl,
                      width: 75,
                    ),
            ),
            const SizedBox(
              width: 16,
              height: 16,
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
            const Spacer(),
            fromHive
                ? const Text('NOT SYNCED',
                    style: TextStyle(
                        color: Colors.red, fontWeight: FontWeight.bold))
                : Container()
          ],
        ),
        const Divider(),
      ],
    );
  }
}
