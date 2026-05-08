#define SOUFFLE_GENERATOR_VERSION "2.5"
#include "souffle/CompiledSouffle.h"
#include "souffle/SignalHandler.h"
#include "souffle/SouffleInterface.h"
#include "souffle/datastructure/BTree.h"
#include "souffle/datastructure/BTreeDelete.h"
#include "souffle/io/IOSystem.h"
#include "souffle/utility/MiscUtil.h"
#include <any>
namespace functors {
extern "C" {
}
} //namespace functors
namespace souffle::t_btree_000_iii__2__0_1_2__001__110__111 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 3;
using t_tuple = Tuple<RamDomain, 3>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2])) ? -1 : (ramBitCast<RamSigned>(a[2]) > ramBitCast<RamSigned>(b[2])) ? 1 :(0);
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2]));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2]));
 }
};
using t_ind_0 = btree_multiset<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
struct t_comparator_1{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2])) ? -1 : (ramBitCast<RamSigned>(a[2]) > ramBitCast<RamSigned>(b[2])) ? 1 :(0)));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))|| ((ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1])) && ((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2]))))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]))&&(ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2]));
 }
};
using t_ind_1 = btree_set<t_tuple,t_comparator_1>;
t_ind_1 ind_1;
using iterator = t_ind_1::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
t_ind_1::operation_hints hints_1_lower;
t_ind_1::operation_hints hints_1_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1,RamDomain a2);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_1::iterator> lowerUpperRange_110(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_1::iterator> lowerUpperRange_110(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_1::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_1::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_iii__2__0_1_2__001__110__111 
namespace souffle::t_btree_000_iii__2__0_1_2__001__110__111 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using t_ind_1 = Type::t_ind_1;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_1.insert(t, h.hints_1_lower)) {
ind_0.insert(t, h.hints_0_lower);
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[3];
std::copy(ramDomain, ramDomain + 3, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1,RamDomain a2) {
RamDomain data[3] = {a0,a1,a2};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_1.contains(t, h.hints_1_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_1.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_1.find(t, h.hints_1_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_1.begin(),ind_1.end());
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_1.begin(),ind_1.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_001(lower,upper,h);
}
range<t_ind_1::iterator> Type::lowerUpperRange_110(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_1 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_1.end(), ind_1.end());
}
return make_range(ind_1.lower_bound(lower, h.hints_1_lower), ind_1.upper_bound(upper, h.hints_1_upper));
}
range<t_ind_1::iterator> Type::lowerUpperRange_110(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_110(lower,upper,h);
}
range<t_ind_1::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_1 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_1.find(lower, h.hints_1_lower);
    auto fin = ind_1.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_1.end(), ind_1.end());
}
return make_range(ind_1.lower_bound(lower, h.hints_1_lower), ind_1.upper_bound(upper, h.hints_1_upper));
}
range<t_ind_1::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_111(lower,upper,h);
}
bool Type::empty() const {
return ind_1.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_1.getChunks(400);
}
void Type::purge() {
ind_0.clear();
ind_1.clear();
}
iterator Type::begin() const {
return ind_1.begin();
}
iterator Type::end() const {
return ind_1.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 3 direct b-tree index 0 lex-order [2]\n";
ind_0.printStats(o);
o << " arity 3 direct b-tree index 1 lex-order [0,1,2]\n";
ind_1.printStats(o);
}
} // namespace souffle::t_btree_000_iii__2__0_1_2__001__110__111 
namespace souffle::t_btree_000_iii__2_0_1__001__111 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 3;
using t_tuple = Tuple<RamDomain, 3>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2])) ? -1 : (ramBitCast<RamSigned>(a[2]) > ramBitCast<RamSigned>(b[2])) ? 1 :((ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0)));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2]))|| ((ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2])) && ((ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2]))&&(ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1,RamDomain a2);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_iii__2_0_1__001__111 
namespace souffle::t_btree_000_iii__2_0_1__001__111 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[3];
std::copy(ramDomain, ramDomain + 3, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1,RamDomain a2) {
RamDomain data[3] = {a0,a1,a2};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_001(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_001(lower,upper,h);
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_111(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 3 direct b-tree index 0 lex-order [2,0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_000_iii__2_0_1__001__111 
namespace souffle::t_btree_000_ii__0_1__11 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_ii__0_1__11 
namespace souffle::t_btree_000_ii__0_1__11 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_000_ii__0_1__11 
namespace souffle::t_btree_000_iii__0_1_2__111 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 3;
using t_tuple = Tuple<RamDomain, 3>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2])) ? -1 : (ramBitCast<RamSigned>(a[2]) > ramBitCast<RamSigned>(b[2])) ? 1 :(0)));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))|| ((ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1])) && ((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2]))))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]))&&(ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1,RamDomain a2);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_iii__0_1_2__111 
namespace souffle::t_btree_000_iii__0_1_2__111 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[3];
std::copy(ramDomain, ramDomain + 3, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1,RamDomain a2) {
RamDomain data[3] = {a0,a1,a2};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_111(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 3 direct b-tree index 0 lex-order [0,1,2]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_000_iii__0_1_2__111 
namespace souffle::t_btree_000_ii__0_1__11__10 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_ii__0_1__11__10 
namespace souffle::t_btree_000_ii__0_1__11__10 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_10(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_000_ii__0_1__11__10 
namespace souffle::t_btree_000_i__0__1 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 1;
using t_tuple = Tuple<RamDomain, 1>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :(0);
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_000_i__0__1 
namespace souffle::t_btree_000_i__0__1 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[1];
std::copy(ramDomain, ramDomain + 1, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0) {
RamDomain data[1] = {a0};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_1(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 1 direct b-tree index 0 lex-order [0]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_000_i__0__1 
namespace souffle::t_btree_100_ii__0_1__11__10 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_delete_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool erase(const t_tuple& t);
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_100_ii__0_1__11__10 
namespace souffle::t_btree_100_ii__0_1__11__10 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::erase(const t_tuple& t) {
if (ind_0.erase(t) > 0) {
return true;
} else return false;
}
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_10(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_100_ii__0_1__11__10 
namespace  souffle {
using namespace souffle;
class Stratum_AddView_a31116529c383d0f {
public:
 Stratum_AddView_a31116529c383d0f(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_iii__2_0_1__001__111::Type& rel_delta_AddView_5a77bca8a5713538,t_btree_000_iii__2_0_1__001__111::Type& rel_delta_MulView_2f15444becc9e933,t_btree_000_ii__0_1__11::Type& rel_delta_Sw_add_t1_2a5fc60c68c23492,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_assoc_add_800820a23e042901,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_distrib_4b71e06ebf518377,t_btree_000_ii__0_1__11::Type& rel_delta_Sw_mul_t1_5c7a36dc6697d179,t_btree_000_iii__2_0_1__001__111::Type& rel_new_AddView_f5e550b62a28075f,t_btree_000_iii__2_0_1__001__111::Type& rel_new_MulView_8c512ff623158d76,t_btree_000_ii__0_1__11::Type& rel_new_Sw_add_t1_d6333d1744962121,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_assoc_add_85b78f163109f879,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_assoc_mul_0e174204fb98fc96,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_distrib_ba9d5c78840b2fbe,t_btree_000_ii__0_1__11::Type& rel_new_Sw_mul_t1_4116da89c6273122,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055,t_btree_000_ii__0_1__11::Type& rel_Sw_add_t1_7fbf07e6e5ab6a1e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_add_704b91beb66fed56,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_mul_f1477d2915f00a0e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_distrib_d0c1f339023f111b,t_btree_000_ii__0_1__11::Type& rel_Sw_mul_t1_a2cde5635764f010);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_000_iii__2_0_1__001__111::Type* rel_delta_AddView_5a77bca8a5713538;
t_btree_000_iii__2_0_1__001__111::Type* rel_delta_MulView_2f15444becc9e933;
t_btree_000_ii__0_1__11::Type* rel_delta_Sw_add_t1_2a5fc60c68c23492;
t_btree_000_iii__0_1_2__111::Type* rel_delta_Sw_assoc_add_800820a23e042901;
t_btree_000_iii__0_1_2__111::Type* rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4;
t_btree_000_iii__0_1_2__111::Type* rel_delta_Sw_distrib_4b71e06ebf518377;
t_btree_000_ii__0_1__11::Type* rel_delta_Sw_mul_t1_5c7a36dc6697d179;
t_btree_000_iii__2_0_1__001__111::Type* rel_new_AddView_f5e550b62a28075f;
t_btree_000_iii__2_0_1__001__111::Type* rel_new_MulView_8c512ff623158d76;
t_btree_000_ii__0_1__11::Type* rel_new_Sw_add_t1_d6333d1744962121;
t_btree_000_iii__0_1_2__111::Type* rel_new_Sw_assoc_add_85b78f163109f879;
t_btree_000_iii__0_1_2__111::Type* rel_new_Sw_assoc_mul_0e174204fb98fc96;
t_btree_000_iii__0_1_2__111::Type* rel_new_Sw_distrib_ba9d5c78840b2fbe;
t_btree_000_ii__0_1__11::Type* rel_new_Sw_mul_t1_4116da89c6273122;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_AddView_eee8a986db892f7e;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_MulView_b6381ece37a9f055;
t_btree_000_ii__0_1__11::Type* rel_Sw_add_t1_7fbf07e6e5ab6a1e;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_assoc_add_704b91beb66fed56;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_assoc_mul_f1477d2915f00a0e;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_distrib_d0c1f339023f111b;
t_btree_000_ii__0_1__11::Type* rel_Sw_mul_t1_a2cde5635764f010;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_AddView_a31116529c383d0f::Stratum_AddView_a31116529c383d0f(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_iii__2_0_1__001__111::Type& rel_delta_AddView_5a77bca8a5713538,t_btree_000_iii__2_0_1__001__111::Type& rel_delta_MulView_2f15444becc9e933,t_btree_000_ii__0_1__11::Type& rel_delta_Sw_add_t1_2a5fc60c68c23492,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_assoc_add_800820a23e042901,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4,t_btree_000_iii__0_1_2__111::Type& rel_delta_Sw_distrib_4b71e06ebf518377,t_btree_000_ii__0_1__11::Type& rel_delta_Sw_mul_t1_5c7a36dc6697d179,t_btree_000_iii__2_0_1__001__111::Type& rel_new_AddView_f5e550b62a28075f,t_btree_000_iii__2_0_1__001__111::Type& rel_new_MulView_8c512ff623158d76,t_btree_000_ii__0_1__11::Type& rel_new_Sw_add_t1_d6333d1744962121,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_assoc_add_85b78f163109f879,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_assoc_mul_0e174204fb98fc96,t_btree_000_iii__0_1_2__111::Type& rel_new_Sw_distrib_ba9d5c78840b2fbe,t_btree_000_ii__0_1__11::Type& rel_new_Sw_mul_t1_4116da89c6273122,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055,t_btree_000_ii__0_1__11::Type& rel_Sw_add_t1_7fbf07e6e5ab6a1e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_add_704b91beb66fed56,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_mul_f1477d2915f00a0e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_distrib_d0c1f339023f111b,t_btree_000_ii__0_1__11::Type& rel_Sw_mul_t1_a2cde5635764f010):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_AddView_5a77bca8a5713538(&rel_delta_AddView_5a77bca8a5713538),
rel_delta_MulView_2f15444becc9e933(&rel_delta_MulView_2f15444becc9e933),
rel_delta_Sw_add_t1_2a5fc60c68c23492(&rel_delta_Sw_add_t1_2a5fc60c68c23492),
rel_delta_Sw_assoc_add_800820a23e042901(&rel_delta_Sw_assoc_add_800820a23e042901),
rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4(&rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4),
rel_delta_Sw_distrib_4b71e06ebf518377(&rel_delta_Sw_distrib_4b71e06ebf518377),
rel_delta_Sw_mul_t1_5c7a36dc6697d179(&rel_delta_Sw_mul_t1_5c7a36dc6697d179),
rel_new_AddView_f5e550b62a28075f(&rel_new_AddView_f5e550b62a28075f),
rel_new_MulView_8c512ff623158d76(&rel_new_MulView_8c512ff623158d76),
rel_new_Sw_add_t1_d6333d1744962121(&rel_new_Sw_add_t1_d6333d1744962121),
rel_new_Sw_assoc_add_85b78f163109f879(&rel_new_Sw_assoc_add_85b78f163109f879),
rel_new_Sw_assoc_mul_0e174204fb98fc96(&rel_new_Sw_assoc_mul_0e174204fb98fc96),
rel_new_Sw_distrib_ba9d5c78840b2fbe(&rel_new_Sw_distrib_ba9d5c78840b2fbe),
rel_new_Sw_mul_t1_4116da89c6273122(&rel_new_Sw_mul_t1_4116da89c6273122),
rel_AddView_eee8a986db892f7e(&rel_AddView_eee8a986db892f7e),
rel_MulView_b6381ece37a9f055(&rel_MulView_b6381ece37a9f055),
rel_Sw_add_t1_7fbf07e6e5ab6a1e(&rel_Sw_add_t1_7fbf07e6e5ab6a1e),
rel_Sw_assoc_add_704b91beb66fed56(&rel_Sw_assoc_add_704b91beb66fed56),
rel_Sw_assoc_mul_f1477d2915f00a0e(&rel_Sw_assoc_mul_f1477d2915f00a0e),
rel_Sw_distrib_d0c1f339023f111b(&rel_Sw_distrib_d0c1f339023f111b),
rel_Sw_mul_t1_a2cde5635764f010(&rel_Sw_mul_t1_a2cde5635764f010){
}

void Stratum_AddView_a31116529c383d0f::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(AddView([0,nil,nil,2],[0,nil,nil,3],[1,[0,nil,nil,2],[0,nil,nil,3],0]).
in file q2-bench.dl [52:1-53:53])_");
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_AddView_eee8a986db892f7e->insert(tuple,READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
}
();signalHandler->setMsg(R"_(AddView([0,nil,nil,1],[1,[0,nil,nil,2],[0,nil,nil,3],0],[1,[0,nil,nil,1],[1,[0,nil,nil,2],[0,nil,nil,3],0],0]).
in file q2-bench.dl [54:1-56:79])_");
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_AddView_eee8a986db892f7e->insert(tuple,READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
}
();signalHandler->setMsg(R"_(AddView([0,nil,nil,6],[0,nil,nil,7],[1,[0,nil,nil,6],[0,nil,nil,7],0]).
in file q2-bench.dl [57:1-58:53])_");
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_AddView_eee8a986db892f7e->insert(tuple,READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
}
();signalHandler->setMsg(R"_(AddView([0,nil,nil,4],[2,[0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],0],[1,[0,nil,nil,4],[2,[0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],0],0]).
in file q2-bench.dl [62:1-64:105])_");
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(4)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(4)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_AddView_eee8a986db892f7e->insert(tuple,READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
}
();signalHandler->setMsg(R"_(MulView([0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],[2,[0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],0]).
in file q2-bench.dl [59:1-61:79])_");
[&](){
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_MulView_b6381ece37a9f055->insert(tuple,READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt));
}
();signalHandler->setMsg(R"_(MulView([1,[0,nil,nil,1],[1,[0,nil,nil,2],[0,nil,nil,3],0],0],[1,[0,nil,nil,4],[2,[0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],0],0],[2,[1,[0,nil,nil,1],[1,[0,nil,nil,2],[0,nil,nil,3],0],0],[1,[0,nil,nil,4],[2,[0,nil,nil,5],[1,[0,nil,nil,6],[0,nil,nil,7],0],0],0],0]).
in file q2-bench.dl [65:1-68:113])_");
[&](){
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(4)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(4)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_MulView_b6381ece37a9f055->insert(tuple,READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt));
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_delta_AddView_5a77bca8a5713538->insert(tuple,READ_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_delta_MulView_2f15444becc9e933->insert(tuple,READ_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_add_t1_2a5fc60c68c23492_op_ctxt,rel_delta_Sw_add_t1_2a5fc60c68c23492->createContext());
CREATE_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt,rel_Sw_add_t1_7fbf07e6e5ab6a1e->createContext());
for(const auto& env0 : *rel_Sw_add_t1_7fbf07e6e5ab6a1e) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_Sw_add_t1_2a5fc60c68c23492->insert(tuple,READ_OP_CONTEXT(rel_delta_Sw_add_t1_2a5fc60c68c23492_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_add_800820a23e042901_op_ctxt,rel_delta_Sw_assoc_add_800820a23e042901->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
for(const auto& env0 : *rel_Sw_assoc_add_704b91beb66fed56) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_delta_Sw_assoc_add_800820a23e042901->insert(tuple,READ_OP_CONTEXT(rel_delta_Sw_assoc_add_800820a23e042901_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4_op_ctxt,rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
for(const auto& env0 : *rel_Sw_assoc_mul_f1477d2915f00a0e) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->insert(tuple,READ_OP_CONTEXT(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_distrib_4b71e06ebf518377_op_ctxt,rel_delta_Sw_distrib_4b71e06ebf518377->createContext());
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
for(const auto& env0 : *rel_Sw_distrib_d0c1f339023f111b) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_delta_Sw_distrib_4b71e06ebf518377->insert(tuple,READ_OP_CONTEXT(rel_delta_Sw_distrib_4b71e06ebf518377_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_mul_t1_5c7a36dc6697d179_op_ctxt,rel_delta_Sw_mul_t1_5c7a36dc6697d179->createContext());
CREATE_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt,rel_Sw_mul_t1_a2cde5635764f010->createContext());
for(const auto& env0 : *rel_Sw_mul_t1_a2cde5635764f010) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_Sw_mul_t1_5c7a36dc6697d179->insert(tuple,READ_OP_CONTEXT(rel_delta_Sw_mul_t1_5c7a36dc6697d179_op_ctxt));
}
}
();auto loop_counter = RamUnsigned(1);
iter = 0;
for(;;) {
signalHandler->setMsg(R"_(AddView(a,b,[1,a,b,0]) :- 
   AddView(a,b,_).
in file q2-bench.dl [33:1-33:49])_");
if(!(rel_delta_AddView_5a77bca8a5713538->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_delta_AddView_5a77bca8a5713538) {
if( !(rel_AddView_eee8a986db892f7e->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_AddView_f5e550b62a28075f->insert(tuple,READ_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(AddView(b,a,[1,b,a,0]) :- 
   Sw_add_t1(a,b).
in file q2-bench.dl [78:1-78:48])_");
if(!(rel_delta_Sw_add_t1_2a5fc60c68c23492->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_add_t1_2a5fc60c68c23492_op_ctxt,rel_delta_Sw_add_t1_2a5fc60c68c23492->createContext());
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_delta_Sw_add_t1_2a5fc60c68c23492) {
if( !(rel_AddView_eee8a986db892f7e->contains(Tuple<RamDomain,3>{{ramBitCast(env0[1]),ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[1]),ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_AddView_f5e550b62a28075f->insert(tuple,READ_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(AddView(a,b,[1,a,b,0]) :- 
   Sw_assoc_add(a,b,_).
in file q2-bench.dl [100:1-100:54])_");
if(!(rel_delta_Sw_assoc_add_800820a23e042901->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_add_800820a23e042901_op_ctxt,rel_delta_Sw_assoc_add_800820a23e042901->createContext());
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_delta_Sw_assoc_add_800820a23e042901) {
if( !(rel_AddView_eee8a986db892f7e->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_AddView_f5e550b62a28075f->insert(tuple,READ_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(AddView([1,a,b,0],c,[1,[1,a,b,0],c,0]) :- 
   Sw_assoc_add(a,b,c).
in file q2-bench.dl [101:1-101:76])_");
if(!(rel_delta_Sw_assoc_add_800820a23e042901->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_add_800820a23e042901_op_ctxt,rel_delta_Sw_assoc_add_800820a23e042901->createContext());
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_delta_Sw_assoc_add_800820a23e042901) {
if( !(rel_AddView_eee8a986db892f7e->contains(Tuple<RamDomain,3>{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_AddView_f5e550b62a28075f->insert(tuple,READ_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(AddView([2,a,b,0],[2,a,c,0],[1,[2,a,b,0],[2,a,c,0],0]) :- 
   Sw_distrib(a,b,c).
in file q2-bench.dl [130:1-130:96])_");
if(!(rel_delta_Sw_distrib_4b71e06ebf518377->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_distrib_4b71e06ebf518377_op_ctxt,rel_delta_Sw_distrib_4b71e06ebf518377->createContext());
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_delta_Sw_distrib_4b71e06ebf518377) {
if( !(rel_AddView_eee8a986db892f7e->contains(Tuple<RamDomain,3>{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_AddView_f5e550b62a28075f->insert(tuple,READ_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView(a,b,[2,a,b,0]) :- 
   MulView(a,b,_).
in file q2-bench.dl [37:1-37:49])_");
if(!(rel_delta_MulView_2f15444becc9e933->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_MulView_2f15444becc9e933) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView(b,a,[2,b,a,0]) :- 
   Sw_mul_t1(a,b).
in file q2-bench.dl [89:1-89:48])_");
if(!(rel_delta_Sw_mul_t1_5c7a36dc6697d179->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_mul_t1_5c7a36dc6697d179_op_ctxt,rel_delta_Sw_mul_t1_5c7a36dc6697d179->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_Sw_mul_t1_5c7a36dc6697d179) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(env0[1]),ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[1]),ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView(a,b,[2,a,b,0]) :- 
   Sw_assoc_mul(a,b,_).
in file q2-bench.dl [114:1-114:54])_");
if(!(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4_op_ctxt,rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView([2,a,b,0],c,[2,[2,a,b,0],c,0]) :- 
   Sw_assoc_mul(a,b,c).
in file q2-bench.dl [115:1-115:76])_");
if(!(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4_op_ctxt,rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView(a,b,[2,a,b,0]) :- 
   Sw_distrib(a,b,_).
in file q2-bench.dl [128:1-128:52])_");
if(!(rel_delta_Sw_distrib_4b71e06ebf518377->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_distrib_4b71e06ebf518377_op_ctxt,rel_delta_Sw_distrib_4b71e06ebf518377->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_Sw_distrib_4b71e06ebf518377) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(MulView(a,c,[2,a,c,0]) :- 
   Sw_distrib(a,_,c).
in file q2-bench.dl [129:1-129:52])_");
if(!(rel_delta_Sw_distrib_4b71e06ebf518377->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_Sw_distrib_4b71e06ebf518377_op_ctxt,rel_delta_Sw_distrib_4b71e06ebf518377->createContext());
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_delta_Sw_distrib_4b71e06ebf518377) {
if( !(rel_MulView_b6381ece37a9f055->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[2]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_new_MulView_8c512ff623158d76->insert(tuple,READ_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(Sw_add_t1(a,b) :- 
   AddView(a,b,_).
in file q2-bench.dl [77:1-77:37])_");
if(!(rel_delta_AddView_5a77bca8a5713538->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_add_t1_d6333d1744962121_op_ctxt,rel_new_Sw_add_t1_d6333d1744962121->createContext());
CREATE_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt,rel_Sw_add_t1_7fbf07e6e5ab6a1e->createContext());
for(const auto& env0 : *rel_delta_AddView_5a77bca8a5713538) {
if( !(rel_Sw_add_t1_7fbf07e6e5ab6a1e->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_new_Sw_add_t1_d6333d1744962121->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_add_t1_d6333d1744962121_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(Sw_assoc_add(a,b,c) :- 
   AddView(a,bc,_),
   AddView(b,c,bc).
in file q2-bench.dl [99:1-99:63])_");
if(!(rel_delta_AddView_5a77bca8a5713538->empty()) && !(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_assoc_add_85b78f163109f879_op_ctxt,rel_new_Sw_assoc_add_85b78f163109f879->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
for(const auto& env0 : *rel_delta_AddView_5a77bca8a5713538) {
auto range = rel_AddView_eee8a986db892f7e->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_assoc_add_704b91beb66fed56->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt))) && !(rel_delta_AddView_5a77bca8a5713538->contains(Tuple<RamDomain,3>{{ramBitCast(env1[0]),ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_assoc_add_85b78f163109f879->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_assoc_add_85b78f163109f879_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_assoc_add(a,b,c) :- 
   AddView(a,bc,_),
   AddView(b,c,bc).
in file q2-bench.dl [99:1-99:63])_");
if(!(rel_AddView_eee8a986db892f7e->empty()) && !(rel_delta_AddView_5a77bca8a5713538->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_assoc_add_85b78f163109f879_op_ctxt,rel_new_Sw_assoc_add_85b78f163109f879->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
auto range = rel_delta_AddView_5a77bca8a5713538->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_assoc_add_704b91beb66fed56->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_assoc_add_85b78f163109f879->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_assoc_add_85b78f163109f879_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_assoc_mul(a,b,c) :- 
   MulView(a,bc,_),
   MulView(b,c,bc).
in file q2-bench.dl [113:1-113:63])_");
if(!(rel_delta_MulView_2f15444becc9e933->empty()) && !(rel_MulView_b6381ece37a9f055->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_assoc_mul_0e174204fb98fc96_op_ctxt,rel_new_Sw_assoc_mul_0e174204fb98fc96->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
for(const auto& env0 : *rel_delta_MulView_2f15444becc9e933) {
auto range = rel_MulView_b6381ece37a9f055->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_assoc_mul_f1477d2915f00a0e->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt))) && !(rel_delta_MulView_2f15444becc9e933->contains(Tuple<RamDomain,3>{{ramBitCast(env1[0]),ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_assoc_mul_0e174204fb98fc96->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_assoc_mul_0e174204fb98fc96_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_assoc_mul(a,b,c) :- 
   MulView(a,bc,_),
   MulView(b,c,bc).
in file q2-bench.dl [113:1-113:63])_");
if(!(rel_MulView_b6381ece37a9f055->empty()) && !(rel_delta_MulView_2f15444becc9e933->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_assoc_mul_0e174204fb98fc96_op_ctxt,rel_new_Sw_assoc_mul_0e174204fb98fc96->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
auto range = rel_delta_MulView_2f15444becc9e933->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_assoc_mul_f1477d2915f00a0e->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_assoc_mul_0e174204fb98fc96->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_assoc_mul_0e174204fb98fc96_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_distrib(a,b,c) :- 
   MulView(a,bc,_),
   AddView(b,c,bc).
in file q2-bench.dl [127:1-127:61])_");
if(!(rel_delta_MulView_2f15444becc9e933->empty()) && !(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_distrib_ba9d5c78840b2fbe_op_ctxt,rel_new_Sw_distrib_ba9d5c78840b2fbe->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
for(const auto& env0 : *rel_delta_MulView_2f15444becc9e933) {
auto range = rel_AddView_eee8a986db892f7e->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_distrib_d0c1f339023f111b->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt))) && !(rel_delta_AddView_5a77bca8a5713538->contains(Tuple<RamDomain,3>{{ramBitCast(env1[0]),ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_distrib_ba9d5c78840b2fbe->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_distrib_ba9d5c78840b2fbe_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_distrib(a,b,c) :- 
   MulView(a,bc,_),
   AddView(b,c,bc).
in file q2-bench.dl [127:1-127:61])_");
if(!(rel_MulView_b6381ece37a9f055->empty()) && !(rel_delta_AddView_5a77bca8a5713538->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt,rel_delta_AddView_5a77bca8a5713538->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_distrib_ba9d5c78840b2fbe_op_ctxt,rel_new_Sw_distrib_ba9d5c78840b2fbe->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
auto range = rel_delta_AddView_5a77bca8a5713538->lowerUpperRange_001(Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast<RamDomain>(MIN_RAM_SIGNED), ramBitCast(env0[1])}},Tuple<RamDomain,3>{{ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast<RamDomain>(MAX_RAM_SIGNED), ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_delta_AddView_5a77bca8a5713538_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_Sw_distrib_d0c1f339023f111b->contains(Tuple<RamDomain,3>{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt)))) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env1[0]),ramBitCast(env1[1])}};
rel_new_Sw_distrib_ba9d5c78840b2fbe->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_distrib_ba9d5c78840b2fbe_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(Sw_mul_t1(a,b) :- 
   MulView(a,b,_).
in file q2-bench.dl [88:1-88:37])_");
if(!(rel_delta_MulView_2f15444becc9e933->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_MulView_2f15444becc9e933_op_ctxt,rel_delta_MulView_2f15444becc9e933->createContext());
CREATE_OP_CONTEXT(rel_new_Sw_mul_t1_4116da89c6273122_op_ctxt,rel_new_Sw_mul_t1_4116da89c6273122->createContext());
CREATE_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt,rel_Sw_mul_t1_a2cde5635764f010->createContext());
for(const auto& env0 : *rel_delta_MulView_2f15444becc9e933) {
if( !(rel_Sw_mul_t1_a2cde5635764f010->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_new_Sw_mul_t1_4116da89c6273122->insert(tuple,READ_OP_CONTEXT(rel_new_Sw_mul_t1_4116da89c6273122_op_ctxt));
}
}
}
();}
if(rel_new_AddView_f5e550b62a28075f->empty() && rel_new_MulView_8c512ff623158d76->empty() && rel_new_Sw_add_t1_d6333d1744962121->empty() && rel_new_Sw_assoc_add_85b78f163109f879->empty() && rel_new_Sw_assoc_mul_0e174204fb98fc96->empty() && rel_new_Sw_distrib_ba9d5c78840b2fbe->empty() && rel_new_Sw_mul_t1_4116da89c6273122->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_AddView_f5e550b62a28075f_op_ctxt,rel_new_AddView_f5e550b62a28075f->createContext());
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
for(const auto& env0 : *rel_new_AddView_f5e550b62a28075f) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_AddView_eee8a986db892f7e->insert(tuple,READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
}
}
();std::swap(rel_delta_AddView_5a77bca8a5713538, rel_new_AddView_f5e550b62a28075f);
rel_new_AddView_f5e550b62a28075f->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_MulView_8c512ff623158d76_op_ctxt,rel_new_MulView_8c512ff623158d76->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_new_MulView_8c512ff623158d76) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_MulView_b6381ece37a9f055->insert(tuple,READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt));
}
}
();std::swap(rel_delta_MulView_2f15444becc9e933, rel_new_MulView_8c512ff623158d76);
rel_new_MulView_8c512ff623158d76->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_Sw_add_t1_d6333d1744962121_op_ctxt,rel_new_Sw_add_t1_d6333d1744962121->createContext());
CREATE_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt,rel_Sw_add_t1_7fbf07e6e5ab6a1e->createContext());
for(const auto& env0 : *rel_new_Sw_add_t1_d6333d1744962121) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_Sw_add_t1_7fbf07e6e5ab6a1e->insert(tuple,READ_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt));
}
}
();std::swap(rel_delta_Sw_add_t1_2a5fc60c68c23492, rel_new_Sw_add_t1_d6333d1744962121);
rel_new_Sw_add_t1_d6333d1744962121->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_Sw_assoc_add_85b78f163109f879_op_ctxt,rel_new_Sw_assoc_add_85b78f163109f879->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
for(const auto& env0 : *rel_new_Sw_assoc_add_85b78f163109f879) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_Sw_assoc_add_704b91beb66fed56->insert(tuple,READ_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt));
}
}
();std::swap(rel_delta_Sw_assoc_add_800820a23e042901, rel_new_Sw_assoc_add_85b78f163109f879);
rel_new_Sw_assoc_add_85b78f163109f879->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_Sw_assoc_mul_0e174204fb98fc96_op_ctxt,rel_new_Sw_assoc_mul_0e174204fb98fc96->createContext());
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
for(const auto& env0 : *rel_new_Sw_assoc_mul_0e174204fb98fc96) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_Sw_assoc_mul_f1477d2915f00a0e->insert(tuple,READ_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt));
}
}
();std::swap(rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4, rel_new_Sw_assoc_mul_0e174204fb98fc96);
rel_new_Sw_assoc_mul_0e174204fb98fc96->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_Sw_distrib_ba9d5c78840b2fbe_op_ctxt,rel_new_Sw_distrib_ba9d5c78840b2fbe->createContext());
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
for(const auto& env0 : *rel_new_Sw_distrib_ba9d5c78840b2fbe) {
Tuple<RamDomain,3> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1]),ramBitCast(env0[2])}};
rel_Sw_distrib_d0c1f339023f111b->insert(tuple,READ_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt));
}
}
();std::swap(rel_delta_Sw_distrib_4b71e06ebf518377, rel_new_Sw_distrib_ba9d5c78840b2fbe);
rel_new_Sw_distrib_ba9d5c78840b2fbe->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_Sw_mul_t1_4116da89c6273122_op_ctxt,rel_new_Sw_mul_t1_4116da89c6273122->createContext());
CREATE_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt,rel_Sw_mul_t1_a2cde5635764f010->createContext());
for(const auto& env0 : *rel_new_Sw_mul_t1_4116da89c6273122) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_Sw_mul_t1_a2cde5635764f010->insert(tuple,READ_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt));
}
}
();std::swap(rel_delta_Sw_mul_t1_5c7a36dc6697d179, rel_new_Sw_mul_t1_4116da89c6273122);
rel_new_Sw_mul_t1_4116da89c6273122->purge();
loop_counter = (ramBitCast<RamUnsigned>(loop_counter) + ramBitCast<RamUnsigned>(RamUnsigned(1)));
iter++;
}
iter = 0;
rel_delta_AddView_5a77bca8a5713538->purge();
rel_new_AddView_f5e550b62a28075f->purge();
rel_delta_MulView_2f15444becc9e933->purge();
rel_new_MulView_8c512ff623158d76->purge();
rel_delta_Sw_add_t1_2a5fc60c68c23492->purge();
rel_new_Sw_add_t1_d6333d1744962121->purge();
rel_delta_Sw_assoc_add_800820a23e042901->purge();
rel_new_Sw_assoc_add_85b78f163109f879->purge();
rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4->purge();
rel_new_Sw_assoc_mul_0e174204fb98fc96->purge();
rel_delta_Sw_distrib_4b71e06ebf518377->purge();
rel_new_Sw_distrib_ba9d5c78840b2fbe->purge();
rel_delta_Sw_mul_t1_5c7a36dc6697d179->purge();
rel_new_Sw_mul_t1_4116da89c6273122->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(a	b	leader)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(AddView)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 3, "params": ["a", "b", "leader"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 3, "types": ["r:Math", "r:Math", "r:Math"]}})_"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_AddView_eee8a986db892f7e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(a	b	leader)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(MulView)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 3, "params": ["a", "b", "leader"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 3, "types": ["r:Math", "r:Math", "r:Math"]}})_"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_MulView_b6381ece37a9f055);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_ConstView_9ebbf39364277e1b {
public:
 Stratum_ConstView_9ebbf39364277e1b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_ii__0_1__11::Type& rel_delta_ConstView_4a36d73eae869150,t_btree_000_ii__0_1__11::Type& rel_new_ConstView_037a061323b9bce5,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_000_ii__0_1__11::Type* rel_delta_ConstView_4a36d73eae869150;
t_btree_000_ii__0_1__11::Type* rel_new_ConstView_037a061323b9bce5;
t_btree_000_ii__0_1__11__10::Type* rel_ConstView_6b59e328fa819e7d;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_ConstView_9ebbf39364277e1b::Stratum_ConstView_9ebbf39364277e1b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_ii__0_1__11::Type& rel_delta_ConstView_4a36d73eae869150,t_btree_000_ii__0_1__11::Type& rel_new_ConstView_037a061323b9bce5,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_ConstView_4a36d73eae869150(&rel_delta_ConstView_4a36d73eae869150),
rel_new_ConstView_037a061323b9bce5(&rel_new_ConstView_037a061323b9bce5),
rel_ConstView_6b59e328fa819e7d(&rel_ConstView_6b59e328fa819e7d){
}

void Stratum_ConstView_9ebbf39364277e1b::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(ConstView(1,[0,nil,nil,1]).
in file q2-bench.dl [42:1-42:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(1)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(2,[0,nil,nil,2]).
in file q2-bench.dl [43:1-43:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(2)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(2)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(3,[0,nil,nil,3]).
in file q2-bench.dl [44:1-44:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(3)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(3)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(4,[0,nil,nil,4]).
in file q2-bench.dl [45:1-45:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(4)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(4)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(5,[0,nil,nil,5]).
in file q2-bench.dl [46:1-46:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(5)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(5)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(6,[0,nil,nil,6]).
in file q2-bench.dl [47:1-47:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(6)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(6)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(7,[0,nil,nil,7]).
in file q2-bench.dl [48:1-48:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(7)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(7)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();signalHandler->setMsg(R"_(ConstView(0,[0,nil,nil,0]).
in file q2-bench.dl [49:1-49:32])_");
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
Tuple<RamDomain,2> tuple{{ramBitCast(RamSigned(0)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_ConstView_4a36d73eae869150_op_ctxt,rel_delta_ConstView_4a36d73eae869150->createContext());
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
for(const auto& env0 : *rel_ConstView_6b59e328fa819e7d) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_ConstView_4a36d73eae869150->insert(tuple,READ_OP_CONTEXT(rel_delta_ConstView_4a36d73eae869150_op_ctxt));
}
}
();auto loop_counter = RamUnsigned(1);
iter = 0;
for(;;) {
signalHandler->setMsg(R"_(ConstView(n,[0,nil,nil,n]) :- 
   ConstView(n,_).
in file q2-bench.dl [29:1-29:51])_");
if(!(rel_delta_ConstView_4a36d73eae869150->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delta_ConstView_4a36d73eae869150_op_ctxt,rel_delta_ConstView_4a36d73eae869150->createContext());
CREATE_OP_CONTEXT(rel_new_ConstView_037a061323b9bce5_op_ctxt,rel_new_ConstView_037a061323b9bce5->createContext());
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
for(const auto& env0 : *rel_delta_ConstView_4a36d73eae869150) {
if( !(rel_ConstView_6b59e328fa819e7d->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(env0[0]))}}
))}},READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(env0[0]))}}
))}};
rel_new_ConstView_037a061323b9bce5->insert(tuple,READ_OP_CONTEXT(rel_new_ConstView_037a061323b9bce5_op_ctxt));
}
}
}
();}
if(rel_new_ConstView_037a061323b9bce5->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_ConstView_037a061323b9bce5_op_ctxt,rel_new_ConstView_037a061323b9bce5->createContext());
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
for(const auto& env0 : *rel_new_ConstView_037a061323b9bce5) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ConstView_6b59e328fa819e7d->insert(tuple,READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
}
}
();std::swap(rel_delta_ConstView_4a36d73eae869150, rel_new_ConstView_037a061323b9bce5);
rel_new_ConstView_037a061323b9bce5->purge();
loop_counter = (ramBitCast<RamUnsigned>(loop_counter) + ramBitCast<RamUnsigned>(RamUnsigned(1)));
iter++;
}
iter = 0;
rel_delta_ConstView_4a36d73eae869150->purge();
rel_new_ConstView_037a061323b9bce5->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_IntroTerm_324b6ce36f6ee59c {
public:
 Stratum_IntroTerm_324b6ce36f6ee59c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d,t_btree_000_i__0__1::Type& rel_IntroTerm_81f8efe8da18455f,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_AddView_eee8a986db892f7e;
t_btree_000_ii__0_1__11__10::Type* rel_ConstView_6b59e328fa819e7d;
t_btree_000_i__0__1::Type* rel_IntroTerm_81f8efe8da18455f;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_MulView_b6381ece37a9f055;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_IntroTerm_324b6ce36f6ee59c::Stratum_IntroTerm_324b6ce36f6ee59c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d,t_btree_000_i__0__1::Type& rel_IntroTerm_81f8efe8da18455f,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_AddView_eee8a986db892f7e(&rel_AddView_eee8a986db892f7e),
rel_ConstView_6b59e328fa819e7d(&rel_ConstView_6b59e328fa819e7d),
rel_IntroTerm_81f8efe8da18455f(&rel_IntroTerm_81f8efe8da18455f),
rel_MulView_b6381ece37a9f055(&rel_MulView_b6381ece37a9f055){
}

void Stratum_IntroTerm_324b6ce36f6ee59c::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(IntroTerm([0,nil,nil,n]) :- 
   ConstView(n,_).
in file q2-bench.dl [28:1-28:48])_");
if(!(rel_ConstView_6b59e328fa819e7d->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
CREATE_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt,rel_IntroTerm_81f8efe8da18455f->createContext());
for(const auto& env0 : *rel_ConstView_6b59e328fa819e7d) {
Tuple<RamDomain,1> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(env0[0]))}}
))}};
rel_IntroTerm_81f8efe8da18455f->insert(tuple,READ_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(IntroTerm([1,a,b,0]) :- 
   AddView(a,b,_).
in file q2-bench.dl [32:1-32:45])_");
if(!(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt,rel_IntroTerm_81f8efe8da18455f->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
Tuple<RamDomain,1> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_IntroTerm_81f8efe8da18455f->insert(tuple,READ_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(IntroTerm([2,a,b,0]) :- 
   MulView(a,b,_).
in file q2-bench.dl [36:1-36:45])_");
if(!(rel_MulView_b6381ece37a9f055->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt,rel_IntroTerm_81f8efe8da18455f->createContext());
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
Tuple<RamDomain,1> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_IntroTerm_81f8efe8da18455f->insert(tuple,READ_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt));
}
}
();}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_UF_14586a7b1a829d84 {
public:
 Stratum_UF_14586a7b1a829d84(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_ii__0_1__11::Type& rel_delete_UF_d9de4bf021d78e22,t_btree_000_ii__0_1__11__10::Type& rel_delta_UF_f42bb675b9e28da8,t_btree_000_ii__0_1__11__10::Type& rel_new_UF_965c03a80fc43938,t_btree_000_ii__0_1__11::Type& rel_reject_UF_38cb56b2f80cbd6e,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d,t_btree_000_i__0__1::Type& rel_IntroTerm_81f8efe8da18455f,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055,t_btree_000_ii__0_1__11::Type& rel_Sw_add_t1_7fbf07e6e5ab6a1e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_add_704b91beb66fed56,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_mul_f1477d2915f00a0e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_distrib_d0c1f339023f111b,t_btree_000_ii__0_1__11::Type& rel_Sw_mul_t1_a2cde5635764f010,t_btree_100_ii__0_1__11__10::Type& rel_UF_d982d69220d52e96);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_000_ii__0_1__11::Type* rel_delete_UF_d9de4bf021d78e22;
t_btree_000_ii__0_1__11__10::Type* rel_delta_UF_f42bb675b9e28da8;
t_btree_000_ii__0_1__11__10::Type* rel_new_UF_965c03a80fc43938;
t_btree_000_ii__0_1__11::Type* rel_reject_UF_38cb56b2f80cbd6e;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_AddView_eee8a986db892f7e;
t_btree_000_ii__0_1__11__10::Type* rel_ConstView_6b59e328fa819e7d;
t_btree_000_i__0__1::Type* rel_IntroTerm_81f8efe8da18455f;
t_btree_000_iii__2__0_1_2__001__110__111::Type* rel_MulView_b6381ece37a9f055;
t_btree_000_ii__0_1__11::Type* rel_Sw_add_t1_7fbf07e6e5ab6a1e;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_assoc_add_704b91beb66fed56;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_assoc_mul_f1477d2915f00a0e;
t_btree_000_iii__0_1_2__111::Type* rel_Sw_distrib_d0c1f339023f111b;
t_btree_000_ii__0_1__11::Type* rel_Sw_mul_t1_a2cde5635764f010;
t_btree_100_ii__0_1__11__10::Type* rel_UF_d982d69220d52e96;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_UF_14586a7b1a829d84::Stratum_UF_14586a7b1a829d84(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_000_ii__0_1__11::Type& rel_delete_UF_d9de4bf021d78e22,t_btree_000_ii__0_1__11__10::Type& rel_delta_UF_f42bb675b9e28da8,t_btree_000_ii__0_1__11__10::Type& rel_new_UF_965c03a80fc43938,t_btree_000_ii__0_1__11::Type& rel_reject_UF_38cb56b2f80cbd6e,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_AddView_eee8a986db892f7e,t_btree_000_ii__0_1__11__10::Type& rel_ConstView_6b59e328fa819e7d,t_btree_000_i__0__1::Type& rel_IntroTerm_81f8efe8da18455f,t_btree_000_iii__2__0_1_2__001__110__111::Type& rel_MulView_b6381ece37a9f055,t_btree_000_ii__0_1__11::Type& rel_Sw_add_t1_7fbf07e6e5ab6a1e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_add_704b91beb66fed56,t_btree_000_iii__0_1_2__111::Type& rel_Sw_assoc_mul_f1477d2915f00a0e,t_btree_000_iii__0_1_2__111::Type& rel_Sw_distrib_d0c1f339023f111b,t_btree_000_ii__0_1__11::Type& rel_Sw_mul_t1_a2cde5635764f010,t_btree_100_ii__0_1__11__10::Type& rel_UF_d982d69220d52e96):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delete_UF_d9de4bf021d78e22(&rel_delete_UF_d9de4bf021d78e22),
rel_delta_UF_f42bb675b9e28da8(&rel_delta_UF_f42bb675b9e28da8),
rel_new_UF_965c03a80fc43938(&rel_new_UF_965c03a80fc43938),
rel_reject_UF_38cb56b2f80cbd6e(&rel_reject_UF_38cb56b2f80cbd6e),
rel_AddView_eee8a986db892f7e(&rel_AddView_eee8a986db892f7e),
rel_ConstView_6b59e328fa819e7d(&rel_ConstView_6b59e328fa819e7d),
rel_IntroTerm_81f8efe8da18455f(&rel_IntroTerm_81f8efe8da18455f),
rel_MulView_b6381ece37a9f055(&rel_MulView_b6381ece37a9f055),
rel_Sw_add_t1_7fbf07e6e5ab6a1e(&rel_Sw_add_t1_7fbf07e6e5ab6a1e),
rel_Sw_assoc_add_704b91beb66fed56(&rel_Sw_assoc_add_704b91beb66fed56),
rel_Sw_assoc_mul_f1477d2915f00a0e(&rel_Sw_assoc_mul_f1477d2915f00a0e),
rel_Sw_distrib_d0c1f339023f111b(&rel_Sw_distrib_d0c1f339023f111b),
rel_Sw_mul_t1_a2cde5635764f010(&rel_Sw_mul_t1_a2cde5635764f010),
rel_UF_d982d69220d52e96(&rel_UF_d982d69220d52e96){
}

void Stratum_UF_14586a7b1a829d84::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(UF(t,t) :- 
   IntroTerm(t).
in file q2-bench.dl [25:1-25:26])_");
if(!(rel_IntroTerm_81f8efe8da18455f->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_IntroTerm_81f8efe8da18455f_op_ctxt,rel_IntroTerm_81f8efe8da18455f->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_IntroTerm_81f8efe8da18455f) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[0])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(UF(o,n) :- 
   AddView(a,b,o),
   AddView(a,b,n),
   o != n,
   ord(o) > ord(n).
in file q2-bench.dl [163:1-163:73])_");
if(!(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
auto range = rel_AddView_eee8a986db892f7e->lowerUpperRange_110(Tuple<RamDomain,3>{{ramBitCast(env0[0]), ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,3>{{ramBitCast(env0[0]), ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamSigned>(env0[2]) > ramBitCast<RamSigned>(env1[2])) && (ramBitCast<RamDomain>(env0[2]) != ramBitCast<RamDomain>(env1[2]))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[2]),ramBitCast(env1[2])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(o,n) :- 
   MulView(a,b,o),
   MulView(a,b,n),
   o != n,
   ord(o) > ord(n).
in file q2-bench.dl [164:1-164:73])_");
if(!(rel_MulView_b6381ece37a9f055->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
auto range = rel_MulView_b6381ece37a9f055->lowerUpperRange_110(Tuple<RamDomain,3>{{ramBitCast(env0[0]), ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,3>{{ramBitCast(env0[0]), ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamSigned>(env0[2]) > ramBitCast<RamSigned>(env1[2])) && (ramBitCast<RamDomain>(env0[2]) != ramBitCast<RamDomain>(env1[2]))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[2]),ramBitCast(env1[2])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(o,n) :- 
   ConstView(c,o),
   ConstView(c,n),
   o != n,
   ord(o) > ord(n).
in file q2-bench.dl [165:1-165:71])_");
if(!(rel_ConstView_6b59e328fa819e7d->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt,rel_ConstView_6b59e328fa819e7d->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_ConstView_6b59e328fa819e7d) {
auto range = rel_ConstView_6b59e328fa819e7d->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ConstView_6b59e328fa819e7d_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamSigned>(env0[1]) > ramBitCast<RamSigned>(env1[1])) && (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1]))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env1[1])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,a,b,0],[1,b,a,0]) :- 
   Sw_add_t1(a,b),
   ord([1,a,b,0]) > ord([1,b,a,0]).
in file q2-bench.dl [79:1-81:33])_");
if(!(rel_Sw_add_t1_7fbf07e6e5ab6a1e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt,rel_Sw_add_t1_7fbf07e6e5ab6a1e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_add_t1_7fbf07e6e5ab6a1e) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,b,a,0],[1,a,b,0]) :- 
   Sw_add_t1(a,b),
   ord([1,b,a,0]) > ord([1,a,b,0]).
in file q2-bench.dl [82:1-84:33])_");
if(!(rel_Sw_add_t1_7fbf07e6e5ab6a1e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_add_t1_7fbf07e6e5ab6a1e_op_ctxt,rel_Sw_add_t1_7fbf07e6e5ab6a1e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_add_t1_7fbf07e6e5ab6a1e) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,a,b,0],[2,b,a,0]) :- 
   Sw_mul_t1(a,b),
   ord([2,a,b,0]) > ord([2,b,a,0]).
in file q2-bench.dl [90:1-92:33])_");
if(!(rel_Sw_mul_t1_a2cde5635764f010->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt,rel_Sw_mul_t1_a2cde5635764f010->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_mul_t1_a2cde5635764f010) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,b,a,0],[2,a,b,0]) :- 
   Sw_mul_t1(a,b),
   ord([2,b,a,0]) > ord([2,a,b,0]).
in file q2-bench.dl [93:1-95:33])_");
if(!(rel_Sw_mul_t1_a2cde5635764f010->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_mul_t1_a2cde5635764f010_op_ctxt,rel_Sw_mul_t1_a2cde5635764f010->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_mul_t1_a2cde5635764f010) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,a,[1,b,c,0],0],[1,[1,a,b,0],c,0]) :- 
   Sw_assoc_add(a,b,c),
   ord([1,a,[1,b,c,0],0]) > ord([1,[1,a,b,0],c,0]).
in file q2-bench.dl [102:1-105:33])_");
if(!(rel_Sw_assoc_add_704b91beb66fed56->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_assoc_add_704b91beb66fed56) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,[1,a,b,0],c,0],[1,a,[1,b,c,0],0]) :- 
   Sw_assoc_add(a,b,c),
   ord([1,[1,a,b,0],c,0]) > ord([1,a,[1,b,c,0],0]).
in file q2-bench.dl [106:1-109:33])_");
if(!(rel_Sw_assoc_add_704b91beb66fed56->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_assoc_add_704b91beb66fed56_op_ctxt,rel_Sw_assoc_add_704b91beb66fed56->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_assoc_add_704b91beb66fed56) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,a,[2,b,c,0],0],[2,[2,a,b,0],c,0]) :- 
   Sw_assoc_mul(a,b,c),
   ord([2,a,[2,b,c,0],0]) > ord([2,[2,a,b,0],c,0]).
in file q2-bench.dl [116:1-119:33])_");
if(!(rel_Sw_assoc_mul_f1477d2915f00a0e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_assoc_mul_f1477d2915f00a0e) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,[2,a,b,0],c,0],[2,a,[2,b,c,0],0]) :- 
   Sw_assoc_mul(a,b,c),
   ord([2,[2,a,b,0],c,0]) > ord([2,a,[2,b,c,0],0]).
in file q2-bench.dl [120:1-123:33])_");
if(!(rel_Sw_assoc_mul_f1477d2915f00a0e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_assoc_mul_f1477d2915f00a0e_op_ctxt,rel_Sw_assoc_mul_f1477d2915f00a0e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_assoc_mul_f1477d2915f00a0e) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,a,[1,b,c,0],0],[1,[2,a,b,0],[2,a,c,0],0]) :- 
   Sw_distrib(a,b,c),
   ord([2,a,[1,b,c,0],0]) > ord([1,[2,a,b,0],[2,a,c,0],0]).
in file q2-bench.dl [131:1-135:33])_");
if(!(rel_Sw_distrib_d0c1f339023f111b->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_distrib_d0c1f339023f111b) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,[2,a,b,0],[2,a,c,0],0],[2,a,[1,b,c,0],0]) :- 
   Sw_distrib(a,b,c),
   ord([1,[2,a,b,0],[2,a,c,0],0]) > ord([2,a,[1,b,c,0],0]).
in file q2-bench.dl [136:1-140:33])_");
if(!(rel_Sw_distrib_d0c1f339023f111b->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_Sw_distrib_d0c1f339023f111b_op_ctxt,rel_Sw_distrib_d0c1f339023f111b->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_Sw_distrib_d0c1f339023f111b) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[1])),ramBitCast(ramBitCast(env0[2])),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
();}
signalHandler->setMsg(R"_(UF([1,a,[0,nil,nil,0],0],a) :- 
   AddView(a,[0,nil,nil,0],_),
   ord([1,a,[0,nil,nil,0],0]) > ord(a).
in file q2-bench.dl [143:1-145:29])_");
if(!(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(env0[0]))) {
RamDomain const ref = env0[1];
if (ref == 0) continue;
const RamDomain *env1 = recordTable.unpack(ref,4);
{
if( (ramBitCast<RamDomain>(env1[3]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[0]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[1]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[2]) == ramBitCast<RamDomain>(RamSigned(0)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[0])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(a,[1,a,[0,nil,nil,0],0]) :- 
   AddView(a,[0,nil,nil,0],_),
   ord(a) > ord([1,a,[0,nil,nil,0],0]).
in file q2-bench.dl [146:1-148:29])_");
if(!(rel_AddView_eee8a986db892f7e->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_AddView_eee8a986db892f7e_op_ctxt,rel_AddView_eee8a986db892f7e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_AddView_eee8a986db892f7e) {
if( (ramBitCast<RamSigned>(env0[0]) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
RamDomain const ref = env0[1];
if (ref == 0) continue;
const RamDomain *env1 = recordTable.unpack(ref,4);
{
if( (ramBitCast<RamDomain>(env1[3]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[0]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[1]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[2]) == ramBitCast<RamDomain>(RamSigned(0)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(1))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
}
();}
signalHandler->setMsg(R"_(UF([2,a,[0,nil,nil,1],0],a) :- 
   MulView(a,[0,nil,nil,1],_),
   ord([2,a,[0,nil,nil,1],0]) > ord(a).
in file q2-bench.dl [151:1-153:29])_");
if(!(rel_MulView_b6381ece37a9f055->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
if( (ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)) > ramBitCast<RamSigned>(env0[0]))) {
RamDomain const ref = env0[1];
if (ref == 0) continue;
const RamDomain *env1 = recordTable.unpack(ref,4);
{
if( (ramBitCast<RamDomain>(env1[3]) == ramBitCast<RamDomain>(RamSigned(1))) && (ramBitCast<RamDomain>(env1[0]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[1]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[2]) == ramBitCast<RamDomain>(RamSigned(0)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)),ramBitCast(env0[0])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(a,[2,a,[0,nil,nil,1],0]) :- 
   MulView(a,[0,nil,nil,1],_),
   ord(a) > ord([2,a,[0,nil,nil,1],0]).
in file q2-bench.dl [154:1-156:29])_");
if(!(rel_MulView_b6381ece37a9f055->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_MulView_b6381ece37a9f055_op_ctxt,rel_MulView_b6381ece37a9f055->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_MulView_b6381ece37a9f055) {
if( (ramBitCast<RamSigned>(env0[0]) > ramBitCast<RamSigned>(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
)))) {
RamDomain const ref = env0[1];
if (ref == 0) continue;
const RamDomain *env1 = recordTable.unpack(ref,4);
{
if( (ramBitCast<RamDomain>(env1[3]) == ramBitCast<RamDomain>(RamSigned(1))) && (ramBitCast<RamDomain>(env1[0]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[1]) == ramBitCast<RamDomain>(RamSigned(0))) && (ramBitCast<RamDomain>(env1[2]) == ramBitCast<RamDomain>(RamSigned(0)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(2))),ramBitCast(ramBitCast(env0[0])),ramBitCast(ramBitCast(pack(recordTable,Tuple<RamDomain,4>{{ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(0))),ramBitCast(ramBitCast(RamSigned(1)))}}
))),ramBitCast(ramBitCast(RamSigned(0)))}}
))}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(a,b) <= UF(a,c) :- 
   UF(b,c),
   b != c.
in file q2-bench.dl [159:1-159:42])_");
if(!(rel_UF_d982d69220d52e96->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt,rel_delete_UF_d9de4bf021d78e22->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_UF_d982d69220d52e96) {
auto range = rel_UF_d982d69220d52e96->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1])) && !((ramBitCast<RamDomain>(env0[1]) == ramBitCast<RamDomain>(env1[1]))) && rel_UF_d982d69220d52e96->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delete_UF_d9de4bf021d78e22->insert(tuple,READ_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt));
}
}
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt,rel_delete_UF_d9de4bf021d78e22->createContext());
for(const auto& env0 : *rel_delete_UF_d9de4bf021d78e22) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_UF_d982d69220d52e96->erase(tuple);
}
}
();rel_delete_UF_d9de4bf021d78e22->purge();
[&](){
CREATE_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt,rel_delta_UF_f42bb675b9e28da8->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_UF_d982d69220d52e96) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_UF_f42bb675b9e28da8->insert(tuple,READ_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt));
}
}
();auto loop_counter = RamUnsigned(1);
iter = 0;
for(;;) {
rel_delta_UF_f42bb675b9e28da8->purge();
signalHandler->setMsg(R"_(UF(a,b) :- 
   UF(a,b),
   UF(a,c),
   UF(b,c),
   b != c.
in file q2-bench.dl [159:1-159:42])_");
if(!(rel_UF_d982d69220d52e96->empty()) && !(rel_new_UF_965c03a80fc43938->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_new_UF_965c03a80fc43938_op_ctxt,rel_new_UF_965c03a80fc43938->createContext());
CREATE_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt,rel_reject_UF_38cb56b2f80cbd6e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_new_UF_965c03a80fc43938) {
auto range = rel_new_UF_965c03a80fc43938->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_new_UF_965c03a80fc43938_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1])) && !((ramBitCast<RamDomain>(env0[1]) == ramBitCast<RamDomain>(env1[1]))) && rel_UF_d982d69220d52e96->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_reject_UF_38cb56b2f80cbd6e->insert(tuple,READ_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(a,b) :- 
   UF(a,b),
   UF(a,c),
   UF(b,c),
   b != c.
in file q2-bench.dl [159:1-159:42])_");
if(!(rel_UF_d982d69220d52e96->empty()) && !(rel_new_UF_965c03a80fc43938->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_new_UF_965c03a80fc43938_op_ctxt,rel_new_UF_965c03a80fc43938->createContext());
CREATE_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt,rel_reject_UF_38cb56b2f80cbd6e->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_new_UF_965c03a80fc43938) {
auto range = rel_UF_d982d69220d52e96->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1])) && rel_UF_d982d69220d52e96->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_reject_UF_38cb56b2f80cbd6e->insert(tuple,READ_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt));
}
}
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt,rel_delta_UF_f42bb675b9e28da8->createContext());
CREATE_OP_CONTEXT(rel_new_UF_965c03a80fc43938_op_ctxt,rel_new_UF_965c03a80fc43938->createContext());
CREATE_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt,rel_reject_UF_38cb56b2f80cbd6e->createContext());
for(const auto& env0 : *rel_new_UF_965c03a80fc43938) {
if( !(rel_reject_UF_38cb56b2f80cbd6e->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_reject_UF_38cb56b2f80cbd6e_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_UF_f42bb675b9e28da8->insert(tuple,READ_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt));
}
}
}
();rel_reject_UF_38cb56b2f80cbd6e->purge();
rel_new_UF_965c03a80fc43938->purge();
signalHandler->setMsg(R"_(UF(a,b) :- 
   UF(a,b),
   UF(a,c),
   UF(b,c),
   b != c.
in file q2-bench.dl [159:1-159:42])_");
if(!(rel_UF_d982d69220d52e96->empty()) && !(rel_delta_UF_f42bb675b9e28da8->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt,rel_delete_UF_d9de4bf021d78e22->createContext());
CREATE_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt,rel_delta_UF_f42bb675b9e28da8->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_UF_d982d69220d52e96) {
auto range = rel_delta_UF_f42bb675b9e28da8->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1])) && rel_UF_d982d69220d52e96->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delete_UF_d9de4bf021d78e22->insert(tuple,READ_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt));
}
}
}
}
();}
signalHandler->setMsg(R"_(UF(a,b) :- 
   UF(a,b),
   UF(a,c),
   UF(b,c),
   b != c.
in file q2-bench.dl [159:1-159:42])_");
if(!(rel_delta_UF_f42bb675b9e28da8->empty()) && !(rel_UF_d982d69220d52e96->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt,rel_delete_UF_d9de4bf021d78e22->createContext());
CREATE_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt,rel_delta_UF_f42bb675b9e28da8->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_UF_d982d69220d52e96) {
auto range = rel_UF_d982d69220d52e96->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
for(const auto& env1 : range) {
if( (ramBitCast<RamDomain>(env0[1]) != ramBitCast<RamDomain>(env1[1])) && !((ramBitCast<RamDomain>(env0[1]) == ramBitCast<RamDomain>(env1[1]))) && rel_delta_UF_f42bb675b9e28da8->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delete_UF_d9de4bf021d78e22->insert(tuple,READ_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt));
}
}
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delete_UF_d9de4bf021d78e22_op_ctxt,rel_delete_UF_d9de4bf021d78e22->createContext());
for(const auto& env0 : *rel_delete_UF_d9de4bf021d78e22) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_UF_d982d69220d52e96->erase(tuple);
}
}
();rel_delete_UF_d9de4bf021d78e22->purge();
if(rel_delta_UF_f42bb675b9e28da8->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_delta_UF_f42bb675b9e28da8_op_ctxt,rel_delta_UF_f42bb675b9e28da8->createContext());
CREATE_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt,rel_UF_d982d69220d52e96->createContext());
for(const auto& env0 : *rel_delta_UF_f42bb675b9e28da8) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_UF_d982d69220d52e96->insert(tuple,READ_OP_CONTEXT(rel_UF_d982d69220d52e96_op_ctxt));
}
}
();loop_counter = (ramBitCast<RamUnsigned>(loop_counter) + ramBitCast<RamUnsigned>(RamUnsigned(1)));
iter++;
}
iter = 0;
rel_delta_UF_f42bb675b9e28da8->purge();
rel_new_UF_965c03a80fc43938->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(child	parent)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(UF)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 2, "params": ["child", "parent"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 2, "types": ["r:Math", "r:Math"]}})_"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_UF_d982d69220d52e96);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_ConstView_6b59e328fa819e7d->purge();
if (pruneImdtRels) rel_IntroTerm_81f8efe8da18455f->purge();
if (pruneImdtRels) rel_Sw_add_t1_7fbf07e6e5ab6a1e->purge();
if (pruneImdtRels) rel_Sw_assoc_add_704b91beb66fed56->purge();
if (pruneImdtRels) rel_Sw_assoc_mul_f1477d2915f00a0e->purge();
if (pruneImdtRels) rel_Sw_distrib_d0c1f339023f111b->purge();
if (pruneImdtRels) rel_Sw_mul_t1_a2cde5635764f010->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Sf_q2_bench_compiled: public SouffleProgram {
public:
 Sf_q2_bench_compiled();
 ~Sf_q2_bench_compiled();
void run();
void runAll(std::string inputDirectoryArg = "",std::string outputDirectoryArg = "",bool performIOArg = true,bool pruneImdtRelsArg = true);
void printAll([[maybe_unused]] std::string outputDirectoryArg = "");
void loadAll([[maybe_unused]] std::string inputDirectoryArg = "");
void dumpInputs();
void dumpOutputs();
SymbolTable& getSymbolTable();
RecordTable& getRecordTable();
void setNumThreads(std::size_t numThreadsValue);
void executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret);
private:
void runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg);
SymbolTableImpl symTable;
SpecializedRecordTable<0,4> recordTable;
ConcurrentCache<std::string,std::regex> regexCache;
Own<t_btree_000_iii__2__0_1_2__001__110__111::Type> rel_AddView_eee8a986db892f7e;
souffle::RelationWrapper<t_btree_000_iii__2__0_1_2__001__110__111::Type> wrapper_rel_AddView_eee8a986db892f7e;
Own<t_btree_000_iii__2_0_1__001__111::Type> rel_new_AddView_f5e550b62a28075f;
Own<t_btree_000_iii__2_0_1__001__111::Type> rel_delta_AddView_5a77bca8a5713538;
Own<t_btree_000_iii__2__0_1_2__001__110__111::Type> rel_MulView_b6381ece37a9f055;
souffle::RelationWrapper<t_btree_000_iii__2__0_1_2__001__110__111::Type> wrapper_rel_MulView_b6381ece37a9f055;
Own<t_btree_000_iii__2_0_1__001__111::Type> rel_new_MulView_8c512ff623158d76;
Own<t_btree_000_iii__2_0_1__001__111::Type> rel_delta_MulView_2f15444becc9e933;
Own<t_btree_000_ii__0_1__11::Type> rel_Sw_add_t1_7fbf07e6e5ab6a1e;
souffle::RelationWrapper<t_btree_000_ii__0_1__11::Type> wrapper_rel_Sw_add_t1_7fbf07e6e5ab6a1e;
Own<t_btree_000_ii__0_1__11::Type> rel_new_Sw_add_t1_d6333d1744962121;
Own<t_btree_000_ii__0_1__11::Type> rel_delta_Sw_add_t1_2a5fc60c68c23492;
Own<t_btree_000_iii__0_1_2__111::Type> rel_Sw_assoc_add_704b91beb66fed56;
souffle::RelationWrapper<t_btree_000_iii__0_1_2__111::Type> wrapper_rel_Sw_assoc_add_704b91beb66fed56;
Own<t_btree_000_iii__0_1_2__111::Type> rel_new_Sw_assoc_add_85b78f163109f879;
Own<t_btree_000_iii__0_1_2__111::Type> rel_delta_Sw_assoc_add_800820a23e042901;
Own<t_btree_000_iii__0_1_2__111::Type> rel_Sw_assoc_mul_f1477d2915f00a0e;
souffle::RelationWrapper<t_btree_000_iii__0_1_2__111::Type> wrapper_rel_Sw_assoc_mul_f1477d2915f00a0e;
Own<t_btree_000_iii__0_1_2__111::Type> rel_new_Sw_assoc_mul_0e174204fb98fc96;
Own<t_btree_000_iii__0_1_2__111::Type> rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4;
Own<t_btree_000_iii__0_1_2__111::Type> rel_Sw_distrib_d0c1f339023f111b;
souffle::RelationWrapper<t_btree_000_iii__0_1_2__111::Type> wrapper_rel_Sw_distrib_d0c1f339023f111b;
Own<t_btree_000_iii__0_1_2__111::Type> rel_new_Sw_distrib_ba9d5c78840b2fbe;
Own<t_btree_000_iii__0_1_2__111::Type> rel_delta_Sw_distrib_4b71e06ebf518377;
Own<t_btree_000_ii__0_1__11::Type> rel_Sw_mul_t1_a2cde5635764f010;
souffle::RelationWrapper<t_btree_000_ii__0_1__11::Type> wrapper_rel_Sw_mul_t1_a2cde5635764f010;
Own<t_btree_000_ii__0_1__11::Type> rel_new_Sw_mul_t1_4116da89c6273122;
Own<t_btree_000_ii__0_1__11::Type> rel_delta_Sw_mul_t1_5c7a36dc6697d179;
Own<t_btree_000_ii__0_1__11__10::Type> rel_ConstView_6b59e328fa819e7d;
souffle::RelationWrapper<t_btree_000_ii__0_1__11__10::Type> wrapper_rel_ConstView_6b59e328fa819e7d;
Own<t_btree_000_ii__0_1__11::Type> rel_new_ConstView_037a061323b9bce5;
Own<t_btree_000_ii__0_1__11::Type> rel_delta_ConstView_4a36d73eae869150;
Own<t_btree_000_i__0__1::Type> rel_IntroTerm_81f8efe8da18455f;
souffle::RelationWrapper<t_btree_000_i__0__1::Type> wrapper_rel_IntroTerm_81f8efe8da18455f;
Own<t_btree_100_ii__0_1__11__10::Type> rel_UF_d982d69220d52e96;
souffle::RelationWrapper<t_btree_100_ii__0_1__11__10::Type> wrapper_rel_UF_d982d69220d52e96;
Own<t_btree_000_ii__0_1__11__10::Type> rel_new_UF_965c03a80fc43938;
Own<t_btree_000_ii__0_1__11__10::Type> rel_delta_UF_f42bb675b9e28da8;
Own<t_btree_000_ii__0_1__11::Type> rel_reject_UF_38cb56b2f80cbd6e;
Own<t_btree_000_ii__0_1__11::Type> rel_delete_UF_d9de4bf021d78e22;
Stratum_AddView_a31116529c383d0f stratum_AddView_37d50e25a50e9062;
Stratum_ConstView_9ebbf39364277e1b stratum_ConstView_5d704bc8811f8da0;
Stratum_IntroTerm_324b6ce36f6ee59c stratum_IntroTerm_1307a4426d52ff4b;
Stratum_UF_14586a7b1a829d84 stratum_UF_0edcd3dc580fc3ae;
std::string inputDirectory;
std::string outputDirectory;
SignalHandler* signalHandler{SignalHandler::instance()};
std::atomic<RamDomain> ctr{};
std::atomic<std::size_t> iter{};
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Sf_q2_bench_compiled::Sf_q2_bench_compiled():
symTable(),
recordTable(),
regexCache(),
rel_AddView_eee8a986db892f7e(mk<t_btree_000_iii__2__0_1_2__001__110__111::Type>()),
wrapper_rel_AddView_eee8a986db892f7e(0, *rel_AddView_eee8a986db892f7e, *this, "AddView", std::array<const char *,3>{{"r:Math","r:Math","r:Math"}}, std::array<const char *,3>{{"a","b","leader"}}, 0),
rel_new_AddView_f5e550b62a28075f(mk<t_btree_000_iii__2_0_1__001__111::Type>()),
rel_delta_AddView_5a77bca8a5713538(mk<t_btree_000_iii__2_0_1__001__111::Type>()),
rel_MulView_b6381ece37a9f055(mk<t_btree_000_iii__2__0_1_2__001__110__111::Type>()),
wrapper_rel_MulView_b6381ece37a9f055(1, *rel_MulView_b6381ece37a9f055, *this, "MulView", std::array<const char *,3>{{"r:Math","r:Math","r:Math"}}, std::array<const char *,3>{{"a","b","leader"}}, 0),
rel_new_MulView_8c512ff623158d76(mk<t_btree_000_iii__2_0_1__001__111::Type>()),
rel_delta_MulView_2f15444becc9e933(mk<t_btree_000_iii__2_0_1__001__111::Type>()),
rel_Sw_add_t1_7fbf07e6e5ab6a1e(mk<t_btree_000_ii__0_1__11::Type>()),
wrapper_rel_Sw_add_t1_7fbf07e6e5ab6a1e(2, *rel_Sw_add_t1_7fbf07e6e5ab6a1e, *this, "Sw_add_t1", std::array<const char *,2>{{"r:Math","r:Math"}}, std::array<const char *,2>{{"a","b"}}, 0),
rel_new_Sw_add_t1_d6333d1744962121(mk<t_btree_000_ii__0_1__11::Type>()),
rel_delta_Sw_add_t1_2a5fc60c68c23492(mk<t_btree_000_ii__0_1__11::Type>()),
rel_Sw_assoc_add_704b91beb66fed56(mk<t_btree_000_iii__0_1_2__111::Type>()),
wrapper_rel_Sw_assoc_add_704b91beb66fed56(3, *rel_Sw_assoc_add_704b91beb66fed56, *this, "Sw_assoc_add", std::array<const char *,3>{{"r:Math","r:Math","r:Math"}}, std::array<const char *,3>{{"a","b","c"}}, 0),
rel_new_Sw_assoc_add_85b78f163109f879(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_delta_Sw_assoc_add_800820a23e042901(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_Sw_assoc_mul_f1477d2915f00a0e(mk<t_btree_000_iii__0_1_2__111::Type>()),
wrapper_rel_Sw_assoc_mul_f1477d2915f00a0e(4, *rel_Sw_assoc_mul_f1477d2915f00a0e, *this, "Sw_assoc_mul", std::array<const char *,3>{{"r:Math","r:Math","r:Math"}}, std::array<const char *,3>{{"a","b","c"}}, 0),
rel_new_Sw_assoc_mul_0e174204fb98fc96(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_Sw_distrib_d0c1f339023f111b(mk<t_btree_000_iii__0_1_2__111::Type>()),
wrapper_rel_Sw_distrib_d0c1f339023f111b(5, *rel_Sw_distrib_d0c1f339023f111b, *this, "Sw_distrib", std::array<const char *,3>{{"r:Math","r:Math","r:Math"}}, std::array<const char *,3>{{"a","b","c"}}, 0),
rel_new_Sw_distrib_ba9d5c78840b2fbe(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_delta_Sw_distrib_4b71e06ebf518377(mk<t_btree_000_iii__0_1_2__111::Type>()),
rel_Sw_mul_t1_a2cde5635764f010(mk<t_btree_000_ii__0_1__11::Type>()),
wrapper_rel_Sw_mul_t1_a2cde5635764f010(6, *rel_Sw_mul_t1_a2cde5635764f010, *this, "Sw_mul_t1", std::array<const char *,2>{{"r:Math","r:Math"}}, std::array<const char *,2>{{"a","b"}}, 0),
rel_new_Sw_mul_t1_4116da89c6273122(mk<t_btree_000_ii__0_1__11::Type>()),
rel_delta_Sw_mul_t1_5c7a36dc6697d179(mk<t_btree_000_ii__0_1__11::Type>()),
rel_ConstView_6b59e328fa819e7d(mk<t_btree_000_ii__0_1__11__10::Type>()),
wrapper_rel_ConstView_6b59e328fa819e7d(7, *rel_ConstView_6b59e328fa819e7d, *this, "ConstView", std::array<const char *,2>{{"i:number","r:Math"}}, std::array<const char *,2>{{"n","leader"}}, 0),
rel_new_ConstView_037a061323b9bce5(mk<t_btree_000_ii__0_1__11::Type>()),
rel_delta_ConstView_4a36d73eae869150(mk<t_btree_000_ii__0_1__11::Type>()),
rel_IntroTerm_81f8efe8da18455f(mk<t_btree_000_i__0__1::Type>()),
wrapper_rel_IntroTerm_81f8efe8da18455f(8, *rel_IntroTerm_81f8efe8da18455f, *this, "IntroTerm", std::array<const char *,1>{{"r:Math"}}, std::array<const char *,1>{{"t"}}, 0),
rel_UF_d982d69220d52e96(mk<t_btree_100_ii__0_1__11__10::Type>()),
wrapper_rel_UF_d982d69220d52e96(9, *rel_UF_d982d69220d52e96, *this, "UF", std::array<const char *,2>{{"r:Math","r:Math"}}, std::array<const char *,2>{{"child","parent"}}, 0),
rel_new_UF_965c03a80fc43938(mk<t_btree_000_ii__0_1__11__10::Type>()),
rel_delta_UF_f42bb675b9e28da8(mk<t_btree_000_ii__0_1__11__10::Type>()),
rel_reject_UF_38cb56b2f80cbd6e(mk<t_btree_000_ii__0_1__11::Type>()),
rel_delete_UF_d9de4bf021d78e22(mk<t_btree_000_ii__0_1__11::Type>()),
stratum_AddView_37d50e25a50e9062(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_AddView_5a77bca8a5713538,*rel_delta_MulView_2f15444becc9e933,*rel_delta_Sw_add_t1_2a5fc60c68c23492,*rel_delta_Sw_assoc_add_800820a23e042901,*rel_delta_Sw_assoc_mul_3bf14b2d54a20bf4,*rel_delta_Sw_distrib_4b71e06ebf518377,*rel_delta_Sw_mul_t1_5c7a36dc6697d179,*rel_new_AddView_f5e550b62a28075f,*rel_new_MulView_8c512ff623158d76,*rel_new_Sw_add_t1_d6333d1744962121,*rel_new_Sw_assoc_add_85b78f163109f879,*rel_new_Sw_assoc_mul_0e174204fb98fc96,*rel_new_Sw_distrib_ba9d5c78840b2fbe,*rel_new_Sw_mul_t1_4116da89c6273122,*rel_AddView_eee8a986db892f7e,*rel_MulView_b6381ece37a9f055,*rel_Sw_add_t1_7fbf07e6e5ab6a1e,*rel_Sw_assoc_add_704b91beb66fed56,*rel_Sw_assoc_mul_f1477d2915f00a0e,*rel_Sw_distrib_d0c1f339023f111b,*rel_Sw_mul_t1_a2cde5635764f010),
stratum_ConstView_5d704bc8811f8da0(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_ConstView_4a36d73eae869150,*rel_new_ConstView_037a061323b9bce5,*rel_ConstView_6b59e328fa819e7d),
stratum_IntroTerm_1307a4426d52ff4b(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_AddView_eee8a986db892f7e,*rel_ConstView_6b59e328fa819e7d,*rel_IntroTerm_81f8efe8da18455f,*rel_MulView_b6381ece37a9f055),
stratum_UF_0edcd3dc580fc3ae(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delete_UF_d9de4bf021d78e22,*rel_delta_UF_f42bb675b9e28da8,*rel_new_UF_965c03a80fc43938,*rel_reject_UF_38cb56b2f80cbd6e,*rel_AddView_eee8a986db892f7e,*rel_ConstView_6b59e328fa819e7d,*rel_IntroTerm_81f8efe8da18455f,*rel_MulView_b6381ece37a9f055,*rel_Sw_add_t1_7fbf07e6e5ab6a1e,*rel_Sw_assoc_add_704b91beb66fed56,*rel_Sw_assoc_mul_f1477d2915f00a0e,*rel_Sw_distrib_d0c1f339023f111b,*rel_Sw_mul_t1_a2cde5635764f010,*rel_UF_d982d69220d52e96){
addRelation("AddView", wrapper_rel_AddView_eee8a986db892f7e, false, true);
addRelation("MulView", wrapper_rel_MulView_b6381ece37a9f055, false, true);
addRelation("Sw_add_t1", wrapper_rel_Sw_add_t1_7fbf07e6e5ab6a1e, false, false);
addRelation("Sw_assoc_add", wrapper_rel_Sw_assoc_add_704b91beb66fed56, false, false);
addRelation("Sw_assoc_mul", wrapper_rel_Sw_assoc_mul_f1477d2915f00a0e, false, false);
addRelation("Sw_distrib", wrapper_rel_Sw_distrib_d0c1f339023f111b, false, false);
addRelation("Sw_mul_t1", wrapper_rel_Sw_mul_t1_a2cde5635764f010, false, false);
addRelation("ConstView", wrapper_rel_ConstView_6b59e328fa819e7d, false, false);
addRelation("IntroTerm", wrapper_rel_IntroTerm_81f8efe8da18455f, false, false);
addRelation("UF", wrapper_rel_UF_d982d69220d52e96, false, true);
}

 Sf_q2_bench_compiled::~Sf_q2_bench_compiled(){
}

void Sf_q2_bench_compiled::runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){

    this->inputDirectory  = std::move(inputDirectoryArg);
    this->outputDirectory = std::move(outputDirectoryArg);
    this->performIO       = performIOArg;
    this->pruneImdtRels   = pruneImdtRelsArg;

    // set default threads (in embedded mode)
    // if this is not set, and omp is used, the default omp setting of number of cores is used.
#if defined(_OPENMP)
    if (0 < getNumThreads()) { omp_set_num_threads(static_cast<int>(getNumThreads())); }
#endif

    signalHandler->set();
// -- query evaluation --
{
 std::vector<RamDomain> args, ret;
stratum_AddView_37d50e25a50e9062.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_ConstView_5d704bc8811f8da0.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_IntroTerm_1307a4426d52ff4b.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_UF_0edcd3dc580fc3ae.run(args, ret);
}

// -- relation hint statistics --
signalHandler->reset();
}

void Sf_q2_bench_compiled::run(){
runFunction("", "", false, false);
}

void Sf_q2_bench_compiled::runAll(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){
runFunction(inputDirectoryArg, outputDirectoryArg, performIOArg, pruneImdtRelsArg);
}

void Sf_q2_bench_compiled::printAll([[maybe_unused]] std::string outputDirectoryArg){
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(a	b	leader)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(MulView)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 3, "params": ["a", "b", "leader"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 3, "types": ["r:Math", "r:Math", "r:Math"]}})_"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_MulView_b6381ece37a9f055);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(a	b	leader)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(AddView)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 3, "params": ["a", "b", "leader"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 3, "types": ["r:Math", "r:Math", "r:Math"]}})_"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_AddView_eee8a986db892f7e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{R"_(IO)_",R"_(stdoutprintsize)_"},{R"_(attributeNames)_",R"_(child	parent)_"},{R"_(auxArity)_",R"_(0)_"},{R"_(name)_",R"_(UF)_"},{R"_(operation)_",R"_(printsize)_"},{R"_(params)_",R"_({"records": {"Math": {"arity": 4, "params": ["tag", "a", "b", "n"]}}, "relation": {"arity": 2, "params": ["child", "parent"]}})_"},{R"_(types)_",R"_({"ADTs": {}, "records": {"r:Math": {"arity": 4, "types": ["i:number", "r:Math", "r:Math", "i:number"]}}, "relation": {"arity": 2, "types": ["r:Math", "r:Math"]}})_"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_UF_d982d69220d52e96);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

void Sf_q2_bench_compiled::loadAll([[maybe_unused]] std::string inputDirectoryArg){
}

void Sf_q2_bench_compiled::dumpInputs(){
}

void Sf_q2_bench_compiled::dumpOutputs(){
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "MulView";
rwOperation["types"] = R"_({"relation": {"arity": 3, "auxArity": 0, "types": ["r:Math", "r:Math", "r:Math"]}})_";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_MulView_b6381ece37a9f055);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "AddView";
rwOperation["types"] = R"_({"relation": {"arity": 3, "auxArity": 0, "types": ["r:Math", "r:Math", "r:Math"]}})_";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_AddView_eee8a986db892f7e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "UF";
rwOperation["types"] = R"_({"relation": {"arity": 2, "auxArity": 0, "types": ["r:Math", "r:Math"]}})_";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_UF_d982d69220d52e96);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

SymbolTable& Sf_q2_bench_compiled::getSymbolTable(){
return symTable;
}

RecordTable& Sf_q2_bench_compiled::getRecordTable(){
return recordTable;
}

void Sf_q2_bench_compiled::setNumThreads(std::size_t numThreadsValue){
SouffleProgram::setNumThreads(numThreadsValue);
symTable.setNumLanes(getNumThreads());
recordTable.setNumLanes(getNumThreads());
regexCache.setNumLanes(getNumThreads());
}

void Sf_q2_bench_compiled::executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret){
if (name == "AddView") {
stratum_AddView_37d50e25a50e9062.run(args, ret);
return;}
if (name == "ConstView") {
stratum_ConstView_5d704bc8811f8da0.run(args, ret);
return;}
if (name == "IntroTerm") {
stratum_IntroTerm_1307a4426d52ff4b.run(args, ret);
return;}
if (name == "UF") {
stratum_UF_0edcd3dc580fc3ae.run(args, ret);
return;}
fatal(("unknown subroutine " + name).c_str());
}

} // namespace  souffle
namespace souffle {
SouffleProgram *newInstance_q2_bench_compiled(){return new  souffle::Sf_q2_bench_compiled;}
SymbolTable *getST_q2_bench_compiled(SouffleProgram *p){return &reinterpret_cast<souffle::Sf_q2_bench_compiled*>(p)->getSymbolTable();}
} // namespace souffle

#ifndef __EMBEDDED_SOUFFLE__
#include "souffle/CompiledOptions.h"
int main(int argc, char** argv)
{
try{
souffle::CmdOptions opt(R"_(q2-bench.dl)_",
R"_()_",
R"_()_",
false,
R"_()_",
1);
if (!opt.parse(argc,argv)) return 1;
souffle::Sf_q2_bench_compiled obj;
#if defined(_OPENMP) 
obj.setNumThreads(opt.getNumJobs());

#endif
obj.runAll(opt.getInputFileDir(), opt.getOutputFileDir());
return 0;
} catch(std::exception &e) { souffle::SignalHandler::instance()->error(e.what());}
}
#endif

namespace  souffle {
using namespace souffle;
class factory_Sf_q2_bench_compiled: souffle::ProgramFactory {
public:
souffle::SouffleProgram* newInstance();
 factory_Sf_q2_bench_compiled();
private:
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
souffle::SouffleProgram* factory_Sf_q2_bench_compiled::newInstance(){
return new  souffle::Sf_q2_bench_compiled();
}

 factory_Sf_q2_bench_compiled::factory_Sf_q2_bench_compiled():
souffle::ProgramFactory("q2_bench_compiled"){
}

} // namespace  souffle
namespace souffle {

#ifdef __EMBEDDED_SOUFFLE__
extern "C" {
souffle::factory_Sf_q2_bench_compiled __factory_Sf_q2_bench_compiled_instance;
}
#endif
} // namespace souffle

