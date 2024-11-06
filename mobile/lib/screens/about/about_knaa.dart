import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutKnaa extends StatelessWidget {
  const AboutKnaa({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Kanaa/Kanaa (1).jpg",
      "images/Kanaa/Kanaa (2).jpg",
      "images/Kanaa/Kanaa (3).jpg",
      "images/Kanaa/Kanaa (4).jpg",
      "images/Kanaa/Kanaa (5).jpg",
      "images/Kanaa/Kanaa (6).jpg",
      "images/Kanaa/Kanaa (7).jpg",
      "images/Kanaa/Kanaa (8).jpg",
      "images/Kanaa/Kanaa (9).jpg",
      "images/Kanaa/Kanaa (10).jpg",
      "images/Kanaa/Kanaa (11).jpg",
      "images/Kanaa/Kanaa (12).jpg",
      "images/Kanaa/Kanaa (13).jpg",
      "images/Kanaa/Kanaa (14).jpg",
      "images/Kanaa/Kanaa (15).jpg",
      "images/Kanaa/Kanaa (16).jpg",
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            "Kana'a",
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
                    "Kna'a B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 825-1025",
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
                    "● Fine lines rendered with multiple, overlapping brush strokes.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Overlapping line junctions.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Small ticking lines or elongated dots attached to longer lines.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Pendant triangles or chevrons.",
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
