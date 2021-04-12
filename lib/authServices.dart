import 'package:firebase_auth/firebase_auth.dart';

class authServices {
  final FirebaseAuth auth = FirebaseAuth.instance;

  Future loginUser(String email, String password) async {
    try {
      AuthResult result = await auth.signInWithEmailAndPassword(
          email: email, password: password);
      return result.user;
    } catch (e) {
      print(e.toString());
    }
  }
}