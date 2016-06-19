// Chengfeng Shi
// 1237783
// CSE446

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class DecisionTree {
	public static final int AGE = 0;
	public static final int SALARY = 1;
	
	public static void main(String[] args) {
		List<Person> persons = constructPerson();
		double entropy1 = measureEntropy(persons);
		System.out.println("total entropy " + entropy1);
		// compare with the
		// max IG for age: 0.1445
		double ageThreshold = findThresholdAndConditionalEntropy(persons, AGE, entropy1);
		// max IG for age: 0.3219
		double salaryThreshold = findThresholdAndConditionalEntropy(persons, SALARY, entropy1);
		
		// so we decided to chose first split by salary with threshold 32500
		
		
		List<Person> personsBelowSalaryThreshold = new LinkedList<Person>();
		List<Person> personAboveEqSalaryThreshold = new LinkedList<Person>();
		
		for (Person p: persons) {
			double s = p.getSalary();
			if (s < salaryThreshold) {
				personsBelowSalaryThreshold.add(p);
			} else {
				personAboveEqSalaryThreshold.add(p);
			}
		}
		
		if (personsBelowSalaryThreshold.size() > 1) {
			double entropy2 = measureEntropy(personsBelowSalaryThreshold);
			double salaryThresholdForBelowAgeThreshold = findThresholdAndConditionalEntropy(personsBelowSalaryThreshold, AGE, entropy2);
		} else {
			System.out.println("No need for further split for below salary " + salaryThreshold);
		}
		
		if (personAboveEqSalaryThreshold.size() > 1) {
			double entropy3 = measureEntropy(personAboveEqSalaryThreshold);
			double salaryThresholdForAboveEqAgeThreshold = findThresholdAndConditionalEntropy(personAboveEqSalaryThreshold, AGE, entropy3);
		} else {
			System.out.println("No need for further split");
		}
		
	}
	
	/**
	 * construct the input data
	 */
	public static List<Person> constructPerson() {
		List<Person> persons = new LinkedList<Person>();
		persons.add(new Person(24, 40000, true));
		persons.add(new Person(53, 52000, false));
		persons.add(new Person(23, 25000, false));
		persons.add(new Person(25, 77000, true));
		persons.add(new Person(32, 48000, true));
		persons.add(new Person(52, 110000, true));
		persons.add(new Person(22, 38000, true));
		persons.add(new Person(43, 44000, false));
		persons.add(new Person(52, 27000, false));
		persons.add(new Person(48, 65000, true));
		return persons;
	}
	
	/**
	 *  return the best threshold for the given choice type
	 */
	public static double findThresholdAndConditionalEntropy(List<Person> persons, int choice, double preEntropy) {
		List<Integer> data = new LinkedList<Integer>();
		for (Person p: persons) {
			if (choice == AGE) {
				data.add(p.getAge());
			} else {
				data.add(p.getSalary());
			}
		}
		
		Collections.sort(data);
		
		double maxIG = 0;
		double threshold = -1;
		for (int i = 0; i < data.size() - 1; i++) {
			double curT = (data.get(i) + data.get(i + 1))  * 1.0 / 2;
			double curEntropy = measureConditionalEntropy(persons, choice, curT);
			if (maxIG < preEntropy - curEntropy) {
				maxIG = preEntropy - curEntropy;
				threshold = curT;
			}
		}
		
		String type = choice == AGE? "age" : "salary";
		if (threshold == -1) {
			System.out.println("no need for further split");
		} else {
			System.out.println(type + " best threshold is " + threshold + " with maximun IG " + maxIG);
		}		
		
		return threshold;
	}
	
	/**
	 * compute the largest IG and conditional entropy for the given choice and threshold
	 * @param persons input data
	 * @param choice AGE or SALARY
	 * @param threshold 
	 * @return the conditional entropy
	 */
	public static double measureConditionalEntropy(List<Person> persons, int choice, double threshold) {
		int below = 0; // total number of people that below threshold
		int belowHasD = 0; // below threshold but has degree
		int belowNoD = 0; // below threshold, has no degree
		int aboveEq = 0; // total number of people that above or equal threshold
		int aboveEqHasD = 0; // above or equal and has degree
		int aboveEqNoD = 0; // above or equal but has no degree
		
		if (choice == AGE) { // based on age
			
			for (Person p: persons) {
				if (p.getAge() < threshold) {
					below++;
					if (p.hasDegree()) {
						belowHasD++;;
					} else {
						belowNoD++;
					}
				} else {
					aboveEq++;
					if (p.hasDegree()) {
						aboveEqHasD++;
					} else {
						aboveEqNoD++;
					}
				}
			}
			
			
		} else if (choice == SALARY) { // based on salary
			for (Person p: persons) {
				if (p.getSalary() < threshold) {
					below++;
					if (p.hasDegree()) {
						belowHasD++;;
					} else {
						belowNoD++;
					}
				} else {
					aboveEq++;
					if (p.hasDegree()) {
						aboveEqHasD++;
					} else {
						aboveEqNoD++;
					}
				}
			}
		} else {
			// invalid situation
			return -1;
		}
		
		double pBelow = below * 1.0 / (below + aboveEq);
		double pAboveEq = aboveEq * 1.0 / (below + aboveEq);
		double pBelowHasD = belowHasD * 1.0 / below;
		double pBelowNoD = belowNoD * 1.0 / below;
		double pAboveEqHasD = aboveEqHasD * 1.0 / aboveEq;
		double pAboveEqNoD = aboveEqNoD * 1.0 / aboveEq;
		
		return -1 * pBelow * (pBelowHasD * log2(pBelowHasD) + pBelowNoD * log2(pBelowNoD)) 
				- pAboveEq * (pAboveEqHasD * log2(pAboveEqHasD) + pAboveEqNoD * log2(pAboveEqNoD));
	}
	
	/**
	 * measure the entropy for the given data
	 * @param persons input data
	 * @return entropy of the given data
	 */
	public static double measureEntropy(List<Person> persons) {
		int hasD = 0;
		int noD = 0;
		for (Person p: persons) {
			if (p.hasDegree()) {
				hasD++;
			} else {
				noD++;
			}
		}
		double pHasD = hasD * 1.0 / (hasD + noD);
		double pNoD = noD * 1.0 / (hasD + noD);
		
		return -1 * pHasD * log2(pHasD) - pNoD * log2(pNoD); 
	}
	
	/**
	 * log function base on 2
	 */
	public static double log2(double x) {
		if (Math.abs(x - 0) < 0.001) return 0;
		return Math.log(x) / Math.log(2);
	}
}
