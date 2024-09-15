import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutTusayan extends StatelessWidget {
  const AboutTusayan({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Tusayan/Tusayan (1).jpg",
      "images/Tusayan/Tusayan (2).jpg",
      "images/Tusayan/Tusayan (3).jpg",
      "images/Tusayan/Tusayan (4).jpg",
      "images/Tusayan/Tusayan (5).jpg",
      "images/Tusayan/Tusayan (6).jpg",
      "images/Tusayan/Tusayan (7).jpg",
      "images/Tusayan/Tusayan (8).jpg",
      "images/Tusayan/Tusayan (9).jpg",
      "images/Tusayan/Tusayan (10).jpg",
      "images/Tusayan/Tusayan (11).jpg",
      "images/Tusayan/Tusayan (12).jpg",
      "images/Tusayan/Tusayan (13).jpg",
      "images/Tusayan/Tusayan (14).jpg",
      "images/Tusayan/Tusayan (15).jpg",
      "images/Tusayan/Tusayan (16).jpg",
      "images/Tusayan/Tusayan (17).jpg",
      "images/Tusayan/Tusayan (18).jpg",
      "images/Tusayan/Tusayan (19).jpg",
      "images/Tusayan/Tusayan (20).jpg",
      "images/Tusayan/Tusayan (21).jpg",
      "images/Tusayan/Tusayan (22).jpg",
      "images/Tusayan/Tusayan (23).jpg",
      "images/Tusayan/Tusayan (24).jpg",
      "images/Tusayan/Tusayan (25).jpg",
      "images/Tusayan/Tusayan (26).jpg",
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            'Tuysayan',
            style: TextStyle(
                fontFamily: 'Uber', fontSize: 60, fontWeight: FontWeight.w700),
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CarouselSlider(
                items: imagePaths.map((imagePath) {
                  return Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: ClipRRect(
                      borderRadius:
                          BorderRadius.circular(5), // Apply rounded corners
                      child: Image.asset(imagePath, fit: BoxFit.cover),
                    ),
                  );
                }).toList(),
                options: CarouselOptions(
                  aspectRatio:
                      1.5, // Ensure aspect ratio is the same for all images
                  enlargeCenterPage: true,
                  enableInfiniteScroll: true,
                  initialPage: 0, // Start from the first page
                  autoPlay: true,
                  enlargeStrategy: CenterPageEnlargeStrategy.height,
                ),
              ),
              const SizedBox(height: 16),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Tusayan B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1220-1290",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Features',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    '● Multiple lines of opposing linear dull barbs/"teeth".',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    '● "Spurred" triangles topped with interlocking rectilinear scrolls.',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    '● Tightly-wrapped curvilinear scrolls.',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    '● Massive geometric elements or thick (> 1 cm) lines, often separated by thin parallel lines.',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
