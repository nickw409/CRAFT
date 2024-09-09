import 'package:craft/provider/login_provider.dart';
import 'package:craft/screens/homepage.dart';
import 'package:craft/screens/user_management/login_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';
import 'package:provider/provider.dart';
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

  Future<void> _logout(BuildContext context) async {
    try {
      // Sign out from Firebase
      await FirebaseAuth.instance.signOut();

      // Clear the user info in the provider
      if (context.mounted) {
        Provider.of<LoginProvider>(context, listen: false).logout();
      }

      // Navigate back to the login screen
      if (context.mounted) {
        Navigator.pushReplacement(
          context,
          PageTransition(
              child: const HomePage(), type: PageTransitionType.fade),
        );
      }
    } catch (e) {
      // Handle any errors during sign-out
      if (context.mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Error: ${e.toString()}')));
      }
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
                  Consumer<LoginProvider>(
                    builder: (context, loginProvider, child) {
                      if (loginProvider.isLoggedIn) {
                        return Column(
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
                              onPressed: () => _logout(context),
                              child: const Text("Sign Out"),
                            ),
                          ],
                        );
                      } else {
                        return Column(
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
                        );
                      }
                    },
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
