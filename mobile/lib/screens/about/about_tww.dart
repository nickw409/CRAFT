import 'package:carousel_slider/carousel_slider.dart';
import 'package:craft/screens/about/about_black_mesa.dart';
import 'package:craft/screens/about/about_flagstaff.dart';
import 'package:craft/screens/about/about_kayenta.dart';
import 'package:craft/screens/about/about_knaa.dart';
import 'package:craft/screens/about/about_sosi.dart';
import 'package:craft/screens/about/about_tusayan.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';

class AboutTww extends StatelessWidget {
  const AboutTww({super.key});

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
                    'Tusayan White Ware',
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
              const ExpansionTile(
                subtitle: Text(
                  'Learn More Tusayan White Ware',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 15,
                  ),
                ),
                title: Text(
                  'About',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 25,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                children: [
                  Text(
                    'Manufactured:',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Lorem ipsum odor amet, consectetuer adipiscing elit.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Distribution:',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Lorem ipsum odor amet, consectetuer adipiscing elit.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Physical Characteristics:',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Paste Color: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Temper: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Surface Treatment: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Paint: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Other White Ware Types In Area:',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Cibola: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Little Colorado: Lorem ipsum odor amet, consectetuer adipiscing elit. Ridiculus convallis aliquet ridiculus eleifend gravida. Finibus est rutrum posuere quisque magna consequat.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16)
                ],
              ),
              ExpansionTile(
                title: const Text(
                  'Types',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 25,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                subtitle: const Text(
                  'Classifications of Tusayan White Ware',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 15,
                  ),
                ),
                children: [
                  Wrap(
                    children: [
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutKnaa(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Kana'a",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutBlackMesa(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Black Mesa",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutSosi(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Sosi",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutFlagstaff(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Flagstaff",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutTusayan(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Tusayan",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutKayenta(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Kayenta",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                ],
              )
            ],
          ),
        ),
      ),
    );
  }
}
