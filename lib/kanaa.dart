import 'package:flutter/material.dart';

class kanaa extends StatelessWidget {
  const kanaa({super.key});

  static final List<String> kanaaAssets = [
    "assets/Kanaa/Kanaa (1).jpg",
    "assets/Kanaa/Kanaa (2).jpg",
    "assets/Kanaa/Kanaa (3).jpg",
    "assets/Kanaa/Kanaa (4).jpg",
    "assets/Kanaa/Kanaa (5).jpg",
    "assets/Kanaa/Kanaa (6).jpg",
    "assets/Kanaa/Kanaa (7).jpg",
    "assets/Kanaa/Kanaa (8).jpg",
    "assets/Kanaa/Kanaa (9).jpg",
    "assets/Kanaa/Kanaa (10).jpg",
    "assets/Kanaa/Kanaa (11).jpg",
    "assets/Kanaa/Kanaa (12).jpg",
    "assets/Kanaa/Kanaa (13).jpg",
    "assets/Kanaa/Kanaa (14).jpg",
    "assets/Kanaa/Kanaa (15).jpg",
    "assets/Kanaa/Kanaa (16).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Kana\'a Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Kana\'a B/W: c. 825-1025\n'
          '\n'
          'Fine lines rendered with multiple, overlapping brush strokes.\n'
          'Overlapping line junctions.\n'
            'Small ticking lines or elongated dots attached to longer lines.\n'
            'Pendant triangles or chevrons.\n'
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
              itemCount: kanaaAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  kanaaAssets[index],
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