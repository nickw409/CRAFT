import 'package:flutter/material.dart';

class about extends StatelessWidget {
  const about({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("About this app"),
        backgroundColor: Colors.blueAccent,
      ),
      body: const Column(
        children: [
          Text('This app was written by Leszek Pawlowicz, Research Faculty, Northern Arizona University.\n'
              'Please report any problems, bugs or issues to leszek.pawlowicz@nau.edu.\n'
              'Please be as specific as possible; include your phone/tablet model info, and a clear description of the problem.\n'
              '\n'
              'Version Alpha 0.10.\n'
              'This app will expire on March 31, 2024. This is done to ensure you have the latest version.\n'
              'Contact me for a link to the latest version. Ultimately, there will be a permanent link to the latest version.\n'
              '\n',

            style: TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.w500,
                fontSize: 16),
          ),
        ],
      ),
    );
  }
}