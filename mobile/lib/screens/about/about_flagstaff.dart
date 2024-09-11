import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutFlagstaff extends StatelessWidget {
  const AboutFlagstaff({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Flagstaff/Flagstaff (1).jpg",
      "images/Flagstaff/Flagstaff (2).jpg",
      "images/Flagstaff/Flagstaff (3).jpg",
      "images/Flagstaff/Flagstaff (4).jpg",
      "images/Flagstaff/Flagstaff (5).jpg",
      "images/Flagstaff/Flagstaff (6).jpg",
      "images/Flagstaff/Flagstaff (7).jpg",
      "images/Flagstaff/Flagstaff (8).jpg",
      "images/Flagstaff/Flagstaff (9).jpg",
      "images/Flagstaff/Flagstaff (10).jpg",
      "images/Flagstaff/Flagstaff (11).jpg",
      "images/Flagstaff/Flagstaff (12).jpg",
      "images/Flagstaff/Flagstaff (13).jpg",
      "images/Flagstaff/Flagstaff (14).jpg",
      "images/Flagstaff/Flagstaff (15).jpg",
      "images/Flagstaff/Flagstaff (16).jpg",
      "images/Flagstaff/Flagstaff (17).jpg",
      "images/Flagstaff/Flagstaff (18).jpg",
      "images/Flagstaff/Flagstaff (19).jpg",
      "images/Flagstaff/Flagstaff (20).jpg",
      "images/Flagstaff/Flagstaff (21).jpg",
      "images/Flagstaff/Flagstaff (22).jpg",
      "images/Flagstaff/Flagstaff (23).jpg",
      "images/Flagstaff/Flagstaff (24).jpg",
      "images/Flagstaff/Flagstaff (25).jpg",
    ];

    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(
                height: 16,
              ),
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16),
                child: FittedBox(
                  fit: BoxFit.contain,
                  child: Text(
                    "Flagstaff",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 60,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
              const SizedBox(
                height: 16,
              ),
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
                    "Flagstaff B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1140-1220",
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
                    "● Rectilinear interlocking scrolls.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Sharply-barbed lines opposing other barbed lines.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Checkerboard patterns with central dots.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Basketweave patterns.",
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
