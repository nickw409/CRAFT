import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutBlackMesa extends StatelessWidget {
  const AboutBlackMesa({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      'images/tww1.jpg',
      'images/tww2.jpg',
      'images/tww3.jpg',
      'images/tww4.jpg',
      'images/tww5.jpg',
    ];

    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              const SizedBox(
                height: 16,
              ),
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16),
                child: FittedBox(
                  fit: BoxFit.contain,
                  child: Text(
                    "Black Mesa",
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
                    "Kna'a B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c 1025-1140",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Characteristics:',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "● Lorem ipsum odor amet, consectetuer adipiscing elit.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Lorem ipsum odor amet, consectetuer adipiscing.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Lorem ipsum odor amet, consectetuer adipiscing elit.",
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
