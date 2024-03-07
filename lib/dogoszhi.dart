import 'package:flutter/material.dart';

class dogoszhi extends StatelessWidget {
  const dogoszhi({super.key});

  static final List<String> dogoszhiAssets = [
    "assets/Dogoszhi/Dogoszhi (1).jpg",
    "assets/Dogoszhi/Dogoszhi (2).jpg",
    "assets/Dogoszhi/Dogoszhi (3).jpg",
    "assets/Dogoszhi/Dogoszhi (4).jpg",
    "assets/Dogoszhi/Dogoszhi (5).jpg",
    "assets/Dogoszhi/Dogoszhi (6).jpg",
    "assets/Dogoszhi/Dogoszhi (7).jpg",
    "assets/Dogoszhi/Dogoszhi (8).jpg",
    "assets/Dogoszhi/Dogoszhi (9).jpg",
    "assets/Dogoszhi/Dogoszhi (10).jpg",
    "assets/Dogoszhi/Dogoszhi (11).jpg",
    "assets/Dogoszhi/Dogoszhi (12).jpg",
    "assets/Dogoszhi/Dogoszhi (13).jpg",
    "assets/Dogoszhi/Dogoszhi (14).jpg",
    "assets/Dogoszhi/Dogoszhi (15).jpg",
    "assets/Dogoszhi/Dogoszhi (16).jpg",
    "assets/Dogoszhi/Dogoszhi (17).jpg",
    "assets/Dogoszhi/Dogoszhi (18).jpg",
    "assets/Dogoszhi/Dogoszhi (19).jpg",
    "assets/Dogoszhi/Dogoszhi (20).jpg",
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Dogoszhi Black On White"),
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        children: [
          const Text('Dogoszhi B/W: c. 1030-1290\n'
              '\n'
              'Parallel lines filled with diagonal hachures, sometimes in curving "wing" or "claw".\n'
              'Sweeping and often interlocking hachure rectilinear and curvilinear patterns.\n'
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
              itemCount: dogoszhiAssets.length,
              itemBuilder: (context, index) {
                return Image.asset(
                  dogoszhiAssets[index],
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