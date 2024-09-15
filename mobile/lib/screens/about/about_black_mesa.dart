import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutBlackMesa extends StatelessWidget {
  const AboutBlackMesa({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Black_Mesa/Black_Mesa (1).jpg",
      "images/Black_Mesa/Black_Mesa (2).jpg",
      "images/Black_Mesa/Black_Mesa (3).jpg",
      "images/Black_Mesa/Black_Mesa (4).jpg",
      "images/Black_Mesa/Black_Mesa (5).jpg",
      "images/Black_Mesa/Black_Mesa (6).jpg",
      "images/Black_Mesa/Black_Mesa (7).jpg",
      "images/Black_Mesa/Black_Mesa (8).jpg",
      "images/Black_Mesa/Black_Mesa (9).jpg",
      "images/Black_Mesa/Black_Mesa (10).jpg",
      "images/Black_Mesa/Black_Mesa (11).jpg",
      "images/Black_Mesa/Black_Mesa (12).jpg",
      "images/Black_Mesa/Black_Mesa (13).jpg",
      "images/Black_Mesa/Black_Mesa (14).jpg",
      "images/Black_Mesa/Black_Mesa (15).jpg",
      "images/Black_Mesa/Black_Mesa (16).jpg",
      "images/Black_Mesa/Black_Mesa (17).jpg",
      "images/Black_Mesa/Black_Mesa (18).jpg",
      "images/Black_Mesa/Black_Mesa (19).jpg",
      "images/Black_Mesa/Black_Mesa (20).jpg",
      "images/Black_Mesa/Black_Mesa (21).jpg",
      "images/Black_Mesa/Black_Mesa (22).jpg",
      "images/Black_Mesa/Black_Mesa (23).jpg",
      "images/Black_Mesa/Black_Mesa (24).jpg",
      "images/Black_Mesa/Black_Mesa (25).jpg",
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            'Black Mesa',
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
                    "Black Mesa B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1025-1140",
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
                    "● Darker/bolder than Kana'a.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Massive geometric elements (triangles, rectangles, diamonds and squares).",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Thick lines and isosceles triangles with pendant dots.",
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
