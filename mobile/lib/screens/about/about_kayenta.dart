import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutKayenta extends StatelessWidget {
  const AboutKayenta({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Kayenta/Kayenta (1).jpg",
      "images/Kayenta/Kayenta (2).jpg",
      "images/Kayenta/Kayenta (3).jpg",
      "images/Kayenta/Kayenta (4).jpg",
      "images/Kayenta/Kayenta (5).jpg",
      "images/Kayenta/Kayenta (6).jpg",
      "images/Kayenta/Kayenta (7).jpg",
      "images/Kayenta/Kayenta (8).jpg",
      "images/Kayenta/Kayenta (9).jpg",
      "images/Kayenta/Kayenta (10).jpg",
      "images/Kayenta/Kayenta (11).jpg",
      "images/Kayenta/Kayenta (12).jpg",
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
                    "Kayenta",
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
                    "Kayenta B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1240-1290",
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
                    "● Distinguished from Tusayan B/W by fine cross-hatching (\"mosquito net\") drawn over other designs.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Massive black designs.",
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
