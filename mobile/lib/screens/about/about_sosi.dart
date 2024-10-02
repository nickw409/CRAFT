import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutSosi extends StatelessWidget {
  const AboutSosi({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Sosi/Sosi (1).jpg",
      "images/Sosi/Sosi (2).jpg",
      "images/Sosi/Sosi (3).jpg",
      "images/Sosi/Sosi (4).jpg",
      "images/Sosi/Sosi (5).jpg",
      "images/Sosi/Sosi (6).jpg",
      "images/Sosi/Sosi (7).jpg",
      "images/Sosi/Sosi (8).jpg",
      "images/Sosi/Sosi (9).jpg",
      "images/Sosi/Sosi (10).jpg",
      "images/Sosi/Sosi (11).jpg",
      "images/Sosi/Sosi (12).jpg",
      "images/Sosi/Sosi (13).jpg",
      "images/Sosi/Sosi (14).jpg",
      "images/Sosi/Sosi (15).jpg",
      "images/Sosi/Sosi (16).jpg",
      "images/Sosi/Sosi (17).jpg",
      "images/Sosi/Sosi (18).jpg",
      "images/Sosi/Sosi (19).jpg",
      "images/Sosi/Sosi (20).jpg",
      "images/Sosi/Sosi (21).jpg",
      "images/Sosi/Sosi (22).jpg",
      "images/Sosi/Sosi (23).jpg",
      "images/Sosi/Sosi (24).jpg",
      "images/Sosi/Sosi (25).jpg",
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            'Sosi',
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
                    "Sosi B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1065-1200",
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
                    "● Medium painted lines that repeatedly fold back on themselves with acute or right angles.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Attached, large barbed elements.",
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
