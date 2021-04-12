import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:english_words/english_words.dart';
import 'camera-screen.dart';
import 'package:firebase_core/firebase_core.dart';

class homepageDemo extends StatefulWidget {
  @override
  _homepageDemoState createState() => _homepageDemoState();
}

class _homepageDemoState extends State<homepageDemo> {
  @override

  Widget _RandomText() {
    final wordPair = WordPair.random();
    return Text(
        wordPair.asPascalCase,
        style: TextStyle(
          fontSize: 17,
          color: Color(0xffbd949e),
        ),
    );
  }

  Widget _ExamsList() {
    return ListTile(
      leading: Icon(
            Icons.menu_book_outlined,
            size: 50,
      ),
      title: Text(
            '30 min, 30 sec',
            style: TextStyle(
              fontSize: 10,
            ),
      ),
      subtitle: _RandomText(),
      trailing: Container(
            height: 27,
            width: 70,
            decoration: BoxDecoration(
                  color: Color(0xffdb9090),
                  borderRadius: BorderRadius.only(
                        topRight:Radius.circular(5),
                        topLeft: Radius.circular(5),
                        bottomLeft: Radius.circular(5),
                        bottomRight: Radius.circular(5),
                        ),
            ),
        child: FlatButton(
            onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => Camera()));
            },
            child: Text(
                'Start',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontFamily: 'sfpro'
                ),
                ),
        ),
      ),
    );
  }

  Widget build(BuildContext context) {
    return Scaffold(

      resizeToAvoidBottomInset: false,
      backgroundColor: Color(0xff9cadce),
      body: Column(
        children: <Widget>[
          Container(
            height: 24,
            width: MediaQuery.of(context).size.width,
            color: Colors.transparent,
          ),

          Container(
            height: 50,
            width: MediaQuery.of(context).size.width,
            child: Row(
              children: <Widget>[
                Container(
                  width: MediaQuery.of(context).size.width / 8,
                  color: Colors.transparent,
                  child: IconButton(
                    icon: Icon(Icons.list),
                    iconSize: 35,
                    color: Colors.white,
                    onPressed: () {},
                  ),
                ),
                Container(
                  width: MediaQuery.of(context).size.width * 6 / 8,
                  color: Colors.transparent
                ),
                Container(
                  width: MediaQuery.of(context).size.width / 8,
                  color: Colors.transparent,
                  child: IconButton(
                    icon: Icon(Icons.notifications),
                    iconSize: 30,
                    color: Colors.white,
                    onPressed: () {},
                  ),
                )
              ],
            ),
          ),

          Container(
            height: MediaQuery.of(context).size.height * 2/5 - 75,
            width: MediaQuery.of(context).size.width,
            color: Color(0xff9cadce),
            child: Center(
              child: Padding(
                  padding: EdgeInsets.only(top: 20, left: 45, right: 25, bottom: 25),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: <Widget>[
                      Container(
                        width: MediaQuery.of(context).size.height / 8,
                        height: MediaQuery.of(context).size.height / 8,
                        decoration: BoxDecoration(
                          color: Colors.transparent,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Colors.white,
                            width: 4,
                          )
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: <Widget>[
                            Text(
                                '7',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 26,
                                  fontWeight: FontWeight.bold,
                                )
                            ),
                            Text(
                                'Exams',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 15,
                                )
                            )
                          ],
                        ),
                      ),
                      Container(
                        width: MediaQuery.of(context).size.width * 3 / 4 - 136,
                        height: MediaQuery.of(context).size.height/8,
                        color: Colors.transparent,
                        padding: EdgeInsets.only(left: 20),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: <Widget>[
                            Text(
                              'Hello,',
                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w500,
                                color: Colors.white,
                                fontFamily: 'sfpro',
                              )
                            ),
                            SizedBox(height: 3),
                            Text(
                                'Peter Man',
                                style: TextStyle(
                                    fontSize: 21,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                    fontFamily: 'sfpro'
                                ),
                            ),
                            SizedBox(height: 3),
                            Text(
                                'Please be honest during the exam',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontFamily: 'sfpro'
                                ),
                            ),
                          ],
                        ),
                      )
                    ],
                  ),
              ),
              ),
            ),

          Container(
            height: MediaQuery.of(context).size.height * 3 / 5,
            width: MediaQuery.of(context).size.width,
            //color: Colors.transparent,
            decoration: new BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(60),
                topRight: Radius.circular(60),
              )
            ),
            child: Column(
              children: <Widget>[
                Padding(
                    padding: EdgeInsets.symmetric(horizontal: 50, vertical: 25),
                    child: Container(
                      width: 250,
                      height: 40,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        boxShadow: [
                          BoxShadow(
                          color: Colors.grey.withOpacity(0.5),
                          spreadRadius: 1,
                          blurRadius: 5,
                          offset: Offset(0, 8), // changes position of shadow
                        ),]
                      ),
                      child: Center(
                        child: TextField(
                            style: TextStyle(
                              fontSize: 15,
                              height: 2.0,
                            ),
                            decoration: InputDecoration(
                                contentPadding: const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
                                border: OutlineInputBorder(),
                                hintText: 'Search'
                          ),
                        ),
                      ),
                    ),
                ),
                Container(
                  width: 325,
                  height: MediaQuery.of(context).size.height * 4/6 - 141,
                  color: Colors.transparent,
                  child: ListView(
                    children: <Widget>[
                      _ExamsList(),
                      _ExamsList(),
                      _ExamsList(),
                      _ExamsList(),
                      _ExamsList(),
                      _ExamsList(),
                      _ExamsList(),
                    ],
                  ),
                )
              ],
            ),
          )
        ],
      )
    );
  }
}
