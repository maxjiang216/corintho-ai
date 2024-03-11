#ifndef ALPHABETA_DTYPES_H
#define ALPHABETA_DTYPES_H

#include <limits>
#include <iostream>

namespace AlphaBeta {

class Value {
public:
    enum class ValueType {
        NORMAL,
        POSITIVE_INFINITY,
        NEGATIVE_INFINITY
    };

    Value(float value = 0.0f) : type_(ValueType::NORMAL), value_(value) {}
    Value(ValueType type) : type_(type), value_(0.0f) {
        if (type == ValueType::POSITIVE_INFINITY) {
            value_ = std::numeric_limits<float>::infinity();
        } else if (type == ValueType::NEGATIVE_INFINITY) {
            value_ = -std::numeric_limits<float>::infinity();
        }
    }

    bool operator<(const Value& other) const;
    bool operator>(const Value& other) const;
    bool operator<=(const Value& other) const;
    bool operator>=(const Value& other) const;
    bool operator==(const Value& other) const;
    bool operator!=(const Value& other) const;
    friend std::ostream& operator<<(std::ostream& os, const Value& value);
    

    Value operator-() const;
    
    
private:
    ValueType type_;
    float value_;
};

} // namespace AlphaBeta

#endif // ALPHABETA_DTYPES_H
