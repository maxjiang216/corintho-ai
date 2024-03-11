#include "alphabeta/dtypes.h"

namespace AlphaBeta {

bool Value::operator<(const Value &other) const {
  if (type_ == ValueType::POSITIVE_INFINITY) {
    return false;
  } else if (type_ == ValueType::NEGATIVE_INFINITY) {
    return true;
  } else {
    return value_ < other.value_;
  }
}

bool Value::operator>(const Value &other) const {
  if (type_ == ValueType::POSITIVE_INFINITY) {
    return true;
  } else if (type_ == ValueType::NEGATIVE_INFINITY) {
    return false;
  } else {
    return value_ > other.value_;
  }
}

bool Value::operator<=(const Value &other) const {
  return !(*this > other);
}

bool Value::operator>=(const Value &other) const {
  return !(*this < other);
}

bool Value::operator==(const Value &other) const {
  if (type_ == other.type_) {
    if (type_ == ValueType::NORMAL) {
      return value_ == other.value_;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

bool Value::operator!=(const Value &other) const {
  return !(*this == other);
}

Value Value::operator-() const {
  if (type_ == ValueType::POSITIVE_INFINITY) {
    return Value(ValueType::NEGATIVE_INFINITY);
  } else if (type_ == ValueType::NEGATIVE_INFINITY) {
    return Value(ValueType::POSITIVE_INFINITY);
  } else {
    return Value(-value_);
  }
}

std::ostream &operator<<(std::ostream &os, const Value &value) {
  if (value.type_ == Value::ValueType::POSITIVE_INFINITY) {
    os << "+INFTY";
  } else if (value.type_ == Value::ValueType::NEGATIVE_INFINITY) {
    os << "-INFTY";
  } else {
    os << value.value_;
  }
  return os;
}

}  // namespace AlphaBeta
