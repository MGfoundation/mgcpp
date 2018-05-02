#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

namespace mgcpp {
struct evaluator {
    template <typename Op>
    inline static auto eval(Op const &);
};
}

#endif // EVALUATOR_HPP
