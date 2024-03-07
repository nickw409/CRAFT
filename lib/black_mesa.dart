import 'package:flutter/material.dart';

class black_mesa extends StatelessWidget {
  const black_mesa({super.key});

  static final List<String> blackmesaAssets = [
    "assets/Black_Mesa/Black_Mesa (1).jpg",
    "assets/Black_Mesa/Black_Mesa (2).jpg",
    "assets/Black_Mesa/Black_Mesa (3).jpg",
    "assets/Black_Mesa/Black_Mesa (4).jpg",
    "assets/Black_Mesa/Black_Mesa (5).jpg",
    "assets/Black_Mesa/Black_Mesa (6).jpg",
    "assets/Black_Mesa/Black_Mesa (7).jpg",
    "assets/Black_Mesa/Black_Mesa (8).jpg",
    "assets/Black_Mesa/Black_Mesa (9).jpg",
    "assets/Black_Mesa/Black_Mesa (10).jpg",
    "assets/Black_Mesa/Black_Mesa (11).jpg",
    "assets/Black_Mesa/Black_Mesa (12).jpg",
    "assets/Black_Mesa/Black_Mesa (13).jpg",
    "assets/Black_Mesa/Black_Mesa (14).jpg",
    "assets/Black_Mesa/Black_Mesa (15).jpg",
    "assets/Black_Mesa/Black_Mesa (16).jpg",
    "assets/Black_Mesa/Black_Mesa (17).jpg",
    "assets/Black_Mesa/Black_Mesa (18).jpg",
    "assets/Black_Mesa/Black_Mesa (19).jpg",
    "assets/Black_Mesa/Black_Mesa (20).jpg",
    "assets/Black_Mesa/Black_Mesa (21).jpg",
    "assets/Black_Mesa/Black_Mesa (22).jpg",
    "assets/Black_Mesa/Black_Mesa (23).jpg",
    "assets/Black_Mesa/Black_Mesa (24).jpg",
    "assets/Black_Mesa/Black_Mesa (25).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Black Mesa Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Black Mesa B/W: c. 1025-1140\n'
              '\n'
              'Darker/bolder than Kana''a.\n'
              'Massive geometric elements (triangles, rectangles, diamonds and squares).\n'
              'Thick lines and isosceles triangles with pendant dots.\n'
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
              itemCount: blackmesaAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  blackmesaAssets[index],
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