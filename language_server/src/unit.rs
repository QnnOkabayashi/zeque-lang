use std::fmt;
use std::marker::PhantomData;
use std::ops;

pub struct Unit<T> {
    pub measure: usize,
    _marker: PhantomData<T>,
}

impl<T> Unit<T> {
    pub const fn new(measure: usize) -> Self {
        Unit {
            measure,
            _marker: PhantomData,
        }
    }

    pub const ZERO: Self = Unit::new(0);
}

impl<T> Copy for Unit<T> {}

impl<T> Clone for Unit<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for Unit<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.measure, f)
    }
}

impl<T> Default for Unit<T> {
    fn default() -> Self {
        Self {
            measure: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> PartialEq for Unit<T> {
    fn eq(&self, other: &Self) -> bool {
        self.measure == other.measure
    }
}

impl<T> PartialOrd for Unit<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.measure.partial_cmp(&other.measure)
    }
}

pub struct Offset<T>(pub Unit<T>);

impl<T> Offset<T> {
    pub const fn new(offset: usize) -> Self {
        Offset(Unit::new(offset))
    }
}

impl<T> Copy for Offset<T> {}

impl<T> Clone for Offset<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Offset<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> fmt::Debug for Offset<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0.measure, f)
    }
}

impl<T> Default for Offset<T> {
    fn default() -> Self {
        Offset(Unit::default())
    }
}

impl<T> ops::Add<Offset<T>> for Unit<T> {
    type Output = Unit<T>;

    fn add(self, rhs: Offset<T>) -> Self::Output {
        Unit::new(self.measure + rhs.0.measure)
    }
}

impl<T> ops::Add<Offset<T>> for Offset<T> {
    type Output = Offset<T>;

    fn add(self, rhs: Offset<T>) -> Self::Output {
        Offset(self.0 + rhs)
    }
}

impl<T> ops::AddAssign<Offset<T>> for Unit<T> {
    fn add_assign(&mut self, rhs: Offset<T>) {
        self.measure += rhs.0.measure;
    }
}

impl<T> ops::AddAssign<Offset<T>> for Offset<T> {
    fn add_assign(&mut self, rhs: Offset<T>) {
        self.0 += rhs;
    }
}

impl<T> ops::Sub<Offset<T>> for Unit<T> {
    type Output = Unit<T>;

    fn sub(self, rhs: Offset<T>) -> Self::Output {
        Unit::new(self.measure - rhs.0.measure)
    }
}

impl<T> ops::Sub<Unit<T>> for Unit<T> {
    type Output = Offset<T>;

    fn sub(self, rhs: Unit<T>) -> Self::Output {
        Offset(Unit::new(self.measure - rhs.measure))
    }
}

impl<T> ops::Sub<Offset<T>> for Offset<T> {
    type Output = Offset<T>;

    fn sub(self, rhs: Offset<T>) -> Self::Output {
        Offset(self.0 - rhs)
    }
}

impl<T> ops::SubAssign<Offset<T>> for Unit<T> {
    fn sub_assign(&mut self, rhs: Offset<T>) {
        self.measure -= rhs.0.measure;
    }
}

impl<T> ops::SubAssign<Offset<T>> for Offset<T> {
    fn sub_assign(&mut self, rhs: Offset<T>) {
        self.0 -= rhs;
    }
}

pub trait MakeUnit: Sized {
    fn unit(measure: usize) -> Unit<Self> {
        Unit::new(measure)
    }

    fn offset(offset: usize) -> Offset<Self> {
        Offset(Self::unit(offset))
    }
}

impl<T> MakeUnit for T {}
