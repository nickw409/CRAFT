import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

class AboutTww extends StatelessWidget {
  const AboutTww({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
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
              items: [
                Image.asset('images/tww1.jpg'),
                Image.asset('images/tww2.jpg'),
                Image.asset('images/tww3.jpg'),
                Image.asset('images/tww4.jpg'),
                Image.asset('images/tww5.jpg'),
              ],
              options: CarouselOptions(
                aspectRatio: 1.5,
                enlargeCenterPage: true,
                enableInfiniteScroll: true,
                initialPage: 5,
                autoPlay: true,
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}
