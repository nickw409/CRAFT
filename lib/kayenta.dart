import 'package:flutter/material.dart';

class kayenta extends StatelessWidget {
  const kayenta({super.key});

  static final List<String> kayentaAssets = [
    "assets/Kayenta/Kayenta (1).jpg",
    "assets/Kayenta/Kayenta (2).jpg",
    "assets/Kayenta/Kayenta (3).jpg",
    "assets/Kayenta/Kayenta (4).jpg",
    "assets/Kayenta/Kayenta (5).jpg",
    "assets/Kayenta/Kayenta (6).jpg",
    "assets/Kayenta/Kayenta (7).jpg",
    "assets/Kayenta/Kayenta (8).jpg",
    "assets/Kayenta/Kayenta (9).jpg",
    "assets/Kayenta/Kayenta (10).jpg",
    "assets/Kayenta/Kayenta (11).jpg",
    "assets/Kayenta/Kayenta (12).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Kayenta Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Kayenta B/W: c. 1240-1290\n'
              '\n'
              'Distinguished from Tusayan B/W by fine cross-hatching ("mosquito net") drawn over other designs.\n'
              'Massive black designs.\n'
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
              itemCount: kayentaAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  kayentaAssets[index],
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