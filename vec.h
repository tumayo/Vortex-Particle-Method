#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>


//vector / matrix class
class Veci2 {
public:
    int i;
    int j;

    Veci2(int a, int b) : i(a), j(b) {

    };

    Veci2(int a) : i(a), j(a) {

    };

    ~Veci2() {

    };

    static Veci2 zero(void) {
        return Veci2(0);
    };
    static Veci2 one(void) {
        return Veci2(1);
    };


    Veci2 operator+(const Veci2& other) const {
        return Veci2(this->i + other.i, this->j + other.j);
    };
    Veci2 operator-(const Veci2& other) const {
        return Veci2(this->i - other.i, this->j - other.j);
    };
    Veci2 operator*(const float c) const {
        return Veci2(int(float(this->i) * c), int(float(this->j) * c));
    };
    Veci2 operator/(const float c) const {
        return Veci2(int(float(this->i) / c), int(float(this->j) / c));
    };

    Veci2& operator+=(const Veci2& rhs) {
        this->i += rhs.i;
        this->j += rhs.j;
        return *this;
    };

    Veci2& operator-=(const Veci2& rhs) {
        this->i -= rhs.i;
        this->j -= rhs.j;
        return *this;
    };

    Veci2& operator*=(const float rhs) {
        this->i = int(float(this->i) * rhs);
        this->j = int(float(this->j) * rhs);
        return *this;
    };

    Veci2& operator/=(const float rhs) {
        this->i = int(float(this->i) / rhs);
        this->j = int(float(this->j) / rhs);
        return *this;
    };
};

class Vecf2 {
public:
    float x;
    float y;
    Vecf2() : x(0.0f), y(0.0f) {

    };
    Vecf2(float a, float b) : x(a), y(b) {

    };
    Vecf2(float a) : x(a), y(a) {

    };

    ~Vecf2() {

    };

    static Vecf2 zero(void) {
        return Vecf2(0.0f);
    };
    static Vecf2 one(void) {
        return Vecf2(1.0f);
    };

    float dot(const Vecf2& other) const {
        return this->x * other.x + this->y * other.y;
    };

    float norm(void) const {
        return std::sqrt(this->x * this->x + this->y * this->y);
    };

    float norm2(void) const {
        return this->x * this->x + this->y * this->y;
    };

    Vecf2 operator+(const Vecf2& other) const {
        return Vecf2(this->x + other.x, this->y + other.y);
    };

    Vecf2 operator-(const Vecf2& other) const {
        return Vecf2(this->x - other.x, this->y - other.y);
    };

    Vecf2 operator*(const float c) const {
        return Vecf2(this->x * c, this->y * c);
    };
    Vecf2 operator/(const float c) const {
        return Vecf2(this->x / c, this->y / c);
    };

    Vecf2& operator+=(const Vecf2& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    };

    Vecf2& operator-=(const Vecf2& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        return *this;
    };

    Vecf2& operator*=(const float rhs) {
        this->x *= rhs;
        this->y *= rhs;
        return *this;
    };

    Vecf2& operator/=(const float rhs) {
        this->x /= rhs;
        this->y /= rhs;
        return *this;
    };
};
