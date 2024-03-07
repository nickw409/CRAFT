import 'package:flutter/material.dart';

class sosi extends StatelessWidget {
  const sosi({super.key});

  static final List<String> sosiAssets = [
    "assets/Sosi/Sosi (1).jpg",
    "assets/Sosi/Sosi (2).jpg",
    "assets/Sosi/Sosi (3).jpg",
    "assets/Sosi/Sosi (4).jpg",
    "assets/Sosi/Sosi (5).jpg",
    "assets/Sosi/Sosi (6).jpg",
    "assets/Sosi/Sosi (7).jpg",
    "assets/Sosi/Sosi (8).jpg",
    "assets/Sosi/Sosi (9).jpg",
    "assets/Sosi/Sosi (10).jpg",
    "assets/Sosi/Sosi (11).jpg",
    "assets/Sosi/Sosi (12).jpg",
    "assets/Sosi/Sosi (13).jpg",
    "assets/Sosi/Sosi (14).jpg",
    "assets/Sosi/Sosi (15).jpg",
    "assets/Sosi/Sosi (16).jpg",
    "assets/Sosi/Sosi (17).jpg",
    "assets/Sosi/Sosi (18).jpg",
    "assets/Sosi/Sosi (19).jpg",
    "assets/Sosi/Sosi (20).jpg",
    "assets/Sosi/Sosi (21).jpg",
    "assets/Sosi/Sosi (22).jpg",
    "assets/Sosi/Sosi (23).jpg",
    "assets/Sosi/Sosi (24).jpg",
    "assets/Sosi/Sosi (25).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Sosi Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Sosi B/W: c. 1065-1200\n'
              '\n'
              'Medium painted lines that repeatedly fold back on themselves with acute or right angles.\n'
              'Attached, large barbed elements.\n'
              '\n'
              'Scroll through example sherds below.'
              '\n',

            style: TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.w500,
                fontSize: 16),
          ),
          Container(
            height: 250,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.black, width: 2),
            ),
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: sosiAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  sosiAssets[index],
                  width:250,
                  height:250,
                  fit: BoxFit.cover,
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}