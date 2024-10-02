import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:craft/screens/user_management/login_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';

class RegistrationPage extends StatefulWidget {
  const RegistrationPage({super.key});

  @override
  State<RegistrationPage> createState() => _RegistrationPageState();
}

class _RegistrationPageState extends State<RegistrationPage> {
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  String? selectedRole;
  final _auth = FirebaseAuth.instance;
  final _firestore = FirebaseFirestore.instance;

  final _nameFocusNode = FocusNode();
  final _emailFocusNode = FocusNode();
  final _passwordFocusNode = FocusNode();
  final _confirmPasswordFocusNode = FocusNode();

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _nameFocusNode.dispose();
    _emailFocusNode.dispose();
    _passwordFocusNode.dispose();
    _confirmPasswordFocusNode.dispose();
    super.dispose();
  }

  Future<void> _register() async {
    final name = _nameController.text.trim();
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();
    final confirmPassword = _confirmPasswordController.text.trim();

    if (!_validateEmail(email)) {
      _showError("Invalid email address");
      return;
    }

    if (!_validatePassword(password)) {
      _showError("Password must be at least 6 characters long");
      return;
    }

    if (password != confirmPassword) {
      _showError("Passwords do not match");
      return;
    }

    if (selectedRole == null) {
      _showError("Please select a role");
      return;
    }

    try {
      UserCredential userCredential =
          await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      // Save user data in Firestore
      await _firestore.collection('users').doc(userCredential.user!.uid).set({
        'name': name,
        'email': email,
        'role': selectedRole ?? 'User',
      });

      FirebaseAuth.instance.signOut();

      // Navigate to the login page
      if (mounted) {
        Navigator.pushReplacement(
          context,
          PageTransition(
            child: const LoginPage(),
            type: PageTransitionType.fade,
          ),
        );
      }
    } catch (e) {
      _showError(e.toString());
    }
  }

  bool _validateEmail(String email) {
    // Simple email validation check
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
              'Register',
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
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextFormField(
                  focusNode: _nameFocusNode,
                  controller: _nameController,
                  textCapitalization: TextCapitalization.words,
                  keyboardType: TextInputType.name,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    icon: Icon(Icons.person_rounded),
                    hintText: 'Name',
                  ),
                  onFieldSubmitted: (_) {
                    FocusScope.of(context).requestFocus(_emailFocusNode);
                  },
                ),
                const SizedBox(
                  height: 16,
                ),
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
                    FocusScope.of(context).requestFocus(_passwordFocusNode);
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
                    FocusScope.of(context)
                        .requestFocus(_confirmPasswordFocusNode);
                  },
                ),
                const SizedBox(height: 16),
                TextFormField(
                  focusNode: _confirmPasswordFocusNode,
                  controller: _confirmPasswordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    icon: Icon(Icons.lock_rounded),
                    hintText: 'Confirm Password',
                  ),
                  onFieldSubmitted: (_) {
                    FocusScope.of(context).unfocus();
                  },
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
                const SizedBox(height: 32),
                Center(
                  child: Column(
                    children: [
                      FilledButton(
                        onPressed: _register,
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
          ),
        )),
      ),
    );
  }
}
