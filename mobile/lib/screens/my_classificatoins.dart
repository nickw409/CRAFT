import 'package:craft/widgets/classification_item.dart';
import 'package:flutter/material.dart';

class MyClassificatoins extends StatelessWidget {
  const MyClassificatoins({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: 16,
            ),
            Padding(
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
            SizedBox(
              height: 16,
            ),
            Divider(),
            ClassificationItem(
                imageUrl: 'asd.com',
                title: 'Flagstaff',
                description: '11/28/2000'),
            ClassificationItem(
                imageUrl: 'asd.com',
                title: 'Flagstaff',
                description: '11/28/2000'),
            ClassificationItem(
                imageUrl: 'asd.com',
                title: 'Flagstaff',
                description: '11/28/2000')
          ],
        ),
      ),
    );
  }
}
