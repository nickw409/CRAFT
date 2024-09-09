import 'package:craft/screens/login_page.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';

class RegistrationPage extends StatefulWidget {
  const RegistrationPage({super.key});

  @override
  State<RegistrationPage> createState() => _RegistrationPageState();
}

class _RegistrationPageState extends State<RegistrationPage> {
  String? selectedRole;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
          child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(
              height: 16,
            ),
            const FittedBox(
              fit: BoxFit.contain,
              child: Text(
                'Register',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 60,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(
              height: 16,
            ),
            TextFormField(
              keyboardType: TextInputType.name,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                icon: Icon(Icons.person_rounded),
                hintText: 'Name',
              ),
            ),
            const SizedBox(
              height: 16,
            ),
            TextFormField(
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                icon: Icon(Icons.email_rounded),
                hintText: 'Email',
              ),
            ),
            const SizedBox(height: 16),
            TextFormField(
              obscureText: true,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                icon: Icon(Icons.password_rounded),
                hintText: 'Password',
              ),
            ),
            const SizedBox(height: 16),
            TextFormField(
              obscureText: true,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                icon: Icon(Icons.lock_rounded),
                hintText: 'Confirm Password',
              ),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                icon: Icon(Icons.work_rounded),
                hintText: 'Select Role',
              ),
              value: selectedRole,
              onChanged: (newValue) {
                setState(() {
                  selectedRole = newValue;
                });
              },
              items: const [
                DropdownMenuItem(
                  value: 'Hobbyist',
                  child: Text('Hobbyist'),
                ),
                DropdownMenuItem(
                  value: 'Archaeologist',
                  child: Text('Archaeologist'),
                ),
                DropdownMenuItem(
                  value: 'Student',
                  child: Text('Student'),
                ),
              ],
            ),
            const Spacer(),
            Center(
              child: Column(
                children: [
                  FilledButton(
                    onPressed: () {},
                    child: const Text('Register'),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                          context,
                          PageTransition(
                              child: const LoginPage(),
                              type: PageTransitionType.fade));
                    },
                    child: const Text("Login Instead?"),
                  ),
                ],
              ),
            )
          ],
        ),
      )),
    );
  }
}
