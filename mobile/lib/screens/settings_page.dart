import 'package:craft/screens/user_management/login_page.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';
import 'package:url_launcher/url_launcher.dart';

class SettingsPage extends StatelessWidget {
  SettingsPage({super.key});

  final Uri _url = Uri.parse(
      'https://www.ceias.nau.edu/capstone/projects/CS/2024/CRAFT_S24/');

  Future<void> _launchAboutURL() async {
    if (!await launchUrl(_url)) {
      throw Exception('Could not launch $_url');
    }
  }

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
                  Column(
                    // Show Only if signed on
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
                      TextButton(
                          onPressed: () {}, child: const Text("Sign Out")),
                    ],
                  ),
                  Column(
                    children: [
                      const SizedBox(
                        height: 16,
                      ),
                      const SizedBox(height: 16),
                      // const Icon(Icons.login_rounded, size: 90),
                      const Text(
                        "Not Signed In",
                        style: TextStyle(
                          fontFamily: 'Uber',
                          fontSize: 35,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const Text(
                        "Log in to save your classificatoins",
                        style: TextStyle(
                          fontFamily: 'Uber',
                          fontSize: 15,
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                              context,
                              PageTransition(
                                  child: const LoginPage(),
                                  type: PageTransitionType.fade));
                        },
                        child: const Text("Sign In"),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const Spacer(),
            Center(
              child: Column(
                children: [
                  const Text('© 2024 CRAFT All rights reserved.'),
                  TextButton(
                    onPressed: _launchAboutURL,
                    child: const Text("Learn More"),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
