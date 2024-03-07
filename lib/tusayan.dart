import 'package:flutter/material.dart';

class tusayan extends StatelessWidget {
  const tusayan({super.key});

  static final List<String> tusayanAssets = [
    "assets/Tusayan/Tusayan (1).jpg",
    "assets/Tusayan/Tusayan (2).jpg",
    "assets/Tusayan/Tusayan (3).jpg",
    "assets/Tusayan/Tusayan (4).jpg",
    "assets/Tusayan/Tusayan (5).jpg",
    "assets/Tusayan/Tusayan (6).jpg",
    "assets/Tusayan/Tusayan (7).jpg",
    "assets/Tusayan/Tusayan (8).jpg",
    "assets/Tusayan/Tusayan (9).jpg",
    "assets/Tusayan/Tusayan (10).jpg",
    "assets/Tusayan/Tusayan (11).jpg",
    "assets/Tusayan/Tusayan (12).jpg",
    "assets/Tusayan/Tusayan (13).jpg",
    "assets/Tusayan/Tusayan (14).jpg",
    "assets/Tusayan/Tusayan (15).jpg",
    "assets/Tusayan/Tusayan (16).jpg",
    "assets/Tusayan/Tusayan (17).jpg",
    "assets/Tusayan/Tusayan (18).jpg",
    "assets/Tusayan/Tusayan (19).jpg",
    "assets/Tusayan/Tusayan (20).jpg",
    "assets/Tusayan/Tusayan (21).jpg",
    "assets/Tusayan/Tusayan (22).jpg",
    "assets/Tusayan/Tusayan (23).jpg",
    "assets/Tusayan/Tusayan (24).jpg",
    "assets/Tusayan/Tusayan (25).jpg",
    "assets/Tusayan/Tusayan (26).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Tusayan Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Tusayan B/W: c. 1220-1290\n'
              '\n'
              'Multiple lines of opposing linear dull barbs/"teeth".\n'
              '"Spurred" triangles topped with interlocking rectilinear scrolls.\n'
              'Tightly-wrapped curvilinear scrolls.\n'
              'Massive geometric elements or thick (> 1 cm) lines, often separated by thin parallel lines.\n'
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
              itemCount: tusayanAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  tusayanAssets[index],
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