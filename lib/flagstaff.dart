import 'package:flutter/material.dart';

class flagstaff extends StatelessWidget {
  const flagstaff({super.key});

  static final List<String> flagstaffAssets = [
    "assets/Flagstaff/Flagstaff (1).jpg",
    "assets/Flagstaff/Flagstaff (2).jpg",
    "assets/Flagstaff/Flagstaff (3).jpg",
    "assets/Flagstaff/Flagstaff (4).jpg",
    "assets/Flagstaff/Flagstaff (5).jpg",
    "assets/Flagstaff/Flagstaff (6).jpg",
    "assets/Flagstaff/Flagstaff (7).jpg",
    "assets/Flagstaff/Flagstaff (8).jpg",
    "assets/Flagstaff/Flagstaff (9).jpg",
    "assets/Flagstaff/Flagstaff (10).jpg",
    "assets/Flagstaff/Flagstaff (11).jpg",
    "assets/Flagstaff/Flagstaff (12).jpg",
    "assets/Flagstaff/Flagstaff (13).jpg",
    "assets/Flagstaff/Flagstaff (14).jpg",
    "assets/Flagstaff/Flagstaff (15).jpg",
    "assets/Flagstaff/Flagstaff (16).jpg",
    "assets/Flagstaff/Flagstaff (17).jpg",
    "assets/Flagstaff/Flagstaff (18).jpg",
    "assets/Flagstaff/Flagstaff (19).jpg",
    "assets/Flagstaff/Flagstaff (20).jpg",
    "assets/Flagstaff/Flagstaff (21).jpg",
    "assets/Flagstaff/Flagstaff (22).jpg",
    "assets/Flagstaff/Flagstaff (23).jpg",
    "assets/Flagstaff/Flagstaff (24).jpg",
    "assets/Flagstaff/Flagstaff (25).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Flagstaff Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Flagstaff B/W: c. 1140-1220\n'
              '\n'
              'Rectilinear interlocking scrolls.\n'
              'Closely-spaced lines.\n'
              'Sharply-barbed lines opposing other barbed lines.\n'
              'Checkerboard patterns with central dots.\n'
              'Basketweave patterns.                                             \n'
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
              itemCount: flagstaffAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  flagstaffAssets[index],
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