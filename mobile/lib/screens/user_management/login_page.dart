import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/global_variables.dart';
import 'package:craft/provider/login_provider.dart';
import 'package:craft/screens/homepage.dart';
import 'package:craft/screens/user_management/registration_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';
import 'package:provider/provider.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _auth = FirebaseAuth.instance;
  final _firestore = FirebaseFirestore.instance;
  final _emailFocusNode = FocusNode();
  final _passwordFocusNode = FocusNode();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _emailFocusNode.dispose();
    _passwordFocusNode.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    if (!_validateEmail(email)) {
      // Show email validation error
      _showError("Invalid email address");
      return;
    }

    if (!_validatePassword(password)) {
      // Show password validation error
      _showError("Password must be at least 6 characters");
      return;
    }

    try {
      // Firebase Authentication
      final userCredential = await _auth.signInWithEmailAndPassword(
          email: email, password: password);

      // Trigger LoginProvider to update login state

      currentUser = userCredential.user;
      if (userCredential.user != null) {
        // Fetch user details from Firestore
        final userId = userCredential.user!.uid;
        final userDoc = await _firestore.collection('users').doc(userId).get();

        if (userDoc.exists) {
          final fullName = userDoc['name'] ?? 'User';
          final role = userDoc['role'] ?? 'Role';

          if (mounted) {
            final loginProvider =
                Provider.of<LoginProvider>(context, listen: false);
            loginProvider.login(fullName, role);
          }
          // Navigate to the next page
          if (mounted) {
            Navigator.pushReplacement(
              context,
              PageTransition(
                  child: const HomePage(), type: PageTransitionType.fade),
            );
          }
        } else {
          _showError("User data not found in Firestore");
        }
      }
    } catch (e) {
      _showError(e.toString());
    }
  }

  bool _validateEmail(String email) {
    return RegExp(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        .hasMatch(email);
  }

  bool _validatePassword(String password) {
    // Password should be at least 6 characters long
    return password.length >= 6;
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        appBar: AppBar(
          toolbarHeight: 80,
          title: const Padding(
            padding: EdgeInsets.symmetric(horizontal: 8.0),
            child: Text(
              'Login',
              style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 60,
                  fontWeight: FontWeight.w700),
            ),
          ),
        ),
        body: SafeArea(
            child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextFormField(
                focusNode: _emailFocusNode,
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  icon: Icon(Icons.email_rounded),
                  hintText: 'Email',
                ),
                onFieldSubmitted: (_) {
                  _passwordFocusNode.requestFocus();
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                focusNode: _passwordFocusNode,
                controller: _passwordController,
                obscureText: true,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  icon: Icon(Icons.password_rounded),
                  hintText: 'Password',
                ),
                onFieldSubmitted: (_) {
                  _login();
                },
              ),
              const SizedBox(height: 32),
              Center(
                child: Column(
                  children: [
                    FilledButton(
                      onPressed: _login,
                      child: const Text('Login'),
                    ),
                    TextButton(
                      onPressed: () {
                        Navigator.push(
                            context,
                            PageTransition(
                                child: const RegistrationPage(),
                                type: PageTransitionType.fade));
                      },
                      child: const Text("Register Instead?"),
                    ),
                  ],
                ),
              )
            ],
          ),
        )),
      ),
    );
  }
}
