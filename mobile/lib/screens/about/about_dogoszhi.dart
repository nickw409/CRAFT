import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutDogoszhi extends StatelessWidget {
  const AboutDogoszhi({super.key});

  @override
  Widget build(BuildContext context) {
    final List<String> imagePaths = [
      "images/Dogoszhi/Dogoszhi (1).jpg",
      "images/Dogoszhi/Dogoszhi (2).jpg",
      "images/Dogoszhi/Dogoszhi (3).jpg",
      "images/Dogoszhi/Dogoszhi (4).jpg",
      "images/Dogoszhi/Dogoszhi (5).jpg",
      "images/Dogoszhi/Dogoszhi (6).jpg",
      "images/Dogoszhi/Dogoszhi (7).jpg",
      "images/Dogoszhi/Dogoszhi (8).jpg",
      "images/Dogoszhi/Dogoszhi (9).jpg",
      "images/Dogoszhi/Dogoszhi (10).jpg",
      "images/Dogoszhi/Dogoszhi (11).jpg",
      "images/Dogoszhi/Dogoszhi (12).jpg",
      "images/Dogoszhi/Dogoszhi (13).jpg",
      "images/Dogoszhi/Dogoszhi (14).jpg",
      "images/Dogoszhi/Dogoszhi (15).jpg",
      "images/Dogoszhi/Dogoszhi (16).jpg",
      "images/Dogoszhi/Dogoszhi (17).jpg",
      "images/Dogoszhi/Dogoszhi (18).jpg",
      "images/Dogoszhi/Dogoszhi (19).jpg",
      "images/Dogoszhi/Dogoszhi (20).jpg",
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            'Dogoszhi',
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
                    "Dogoszhi B/W",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "c. 1030-1290",
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
                    "● Parallel lines filled with diagonal hachures, sometimes in curving \"wing\" or \"claw\".",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "● Sweeping and often interlocking hachure rectilinear and curvilinear patterns.",
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
