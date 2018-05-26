#ifndef OUTCOME_HPP
#define OUTCOME_HPP

#define BOOST_SYSTEM_NO_DEPRECATED
#include <boost/outcome.hpp>
namespace outcome {
template <class R>
using result = BOOST_OUTCOME_V2_NAMESPACE::std_result<R>;
template <class R>
using outcome = BOOST_OUTCOME_V2_NAMESPACE::std_outcome<R>;
using BOOST_OUTCOME_V2_NAMESPACE::success;
}  // namespace outcome

#endif  // OUTCOME_HPP
