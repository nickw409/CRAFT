import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Column(
              children: [
                SizedBox(
                  height: 16,
                ),
                Padding(
                  padding: EdgeInsets.symmetric(vertical: 16),
                  child: FittedBox(
                    fit: BoxFit.contain,
                    child: Text(
                      'Settings',
                      style: TextStyle(
                        fontFamily: 'Uber',
                        fontSize: 60,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            Center(
              child: Column(
                children: [
                  const SizedBox(
                    height: 16,
                  ),
                  const SizedBox(height: 16),
                  const Icon(Icons.person_rounded, size: 90),
                  const Text(
                    "John Doe",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 35,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const Text(
                    "Archaeologist",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                    ),
                  ),
                  TextButton(onPressed: () {}, child: Text("Sign Out")),
                ],
              ),
            ),
            const Spacer(),
            Center(
              child: Column(
                children: [
                  const Text('© 2024 CRAFT All rights reserved.'),
                  TextButton(onPressed: () {}, child: Text("Learn More")),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
