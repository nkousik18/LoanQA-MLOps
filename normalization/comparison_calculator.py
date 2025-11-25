"""
Comparison calculator for loan metrics
Calculates total cost, effective rate, flexibility score, and generates comparisons
"""
import logging
from typing import Dict, Any, List
import math

logger = logging.getLogger(__name__)


class ComparisonCalculator:
    """Calculator for loan comparison metrics"""
    
    def calculate_metrics(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comparison metrics for a loan
        
        Args:
            loan_data: Normalized loan data
            
        Returns:
            Comparison metrics
        """
        try:
            metrics = {
                "loan_id": loan_data.get("loan_id"),
                "total_cost_estimate": self._calculate_total_cost(loan_data),
                "effective_interest_rate": self._calculate_effective_rate(loan_data),
                "flexibility_score": self._calculate_flexibility_score(loan_data),
                "monthly_emi": self._calculate_emi(loan_data),
                "total_interest_payable": self._calculate_total_interest(loan_data)
            }
            
            logger.info(f"Calculated metrics for loan {loan_data.get('loan_id')}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _calculate_total_cost(self, loan_data: Dict[str, Any]) -> float:
        """Calculate total cost including principal, interest, and fees"""
        principal = loan_data.get("principal_amount", 0)
        total_interest = self._calculate_total_interest(loan_data)
        
        # Sum all fees
        fees = loan_data.get("fees", [])
        total_fees = sum(fee.get("amount", 0) for fee in fees)
        
        # Add processing fee if available
        processing_fee = loan_data.get("processing_fee", 0)
        
        total_cost = principal + total_interest + total_fees + processing_fee
        
        return round(total_cost, 2)
    
    def _calculate_total_interest(self, loan_data: Dict[str, Any]) -> float:
        """Calculate total interest payable"""
        principal = loan_data.get("principal_amount", 0)
        annual_rate = loan_data.get("interest_rate", 0)
        tenure_months = loan_data.get("tenure_months", 0)
        
        if not all([principal, annual_rate, tenure_months]):
            return 0.0
        
        # Calculate using EMI formula
        monthly_rate = annual_rate / 12 / 100
        
        if monthly_rate == 0:
            return 0.0
        
        # Total amount paid
        emi = self._calculate_emi(loan_data)
        total_paid = emi * tenure_months
        
        # Total interest = Total paid - Principal
        total_interest = total_paid - principal
        
        return round(total_interest, 2)
    
    def _calculate_emi(self, loan_data: Dict[str, Any]) -> float:
        """Calculate monthly EMI"""
        principal = loan_data.get("principal_amount", 0)
        annual_rate = loan_data.get("interest_rate", 0)
        tenure_months = loan_data.get("tenure_months", 0)
        
        if not all([principal, annual_rate, tenure_months]):
            return 0.0
        
        monthly_rate = annual_rate / 12 / 100
        
        if monthly_rate == 0:
            # No interest case
            return round(principal / tenure_months, 2)
        
        # EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
        emi = principal * monthly_rate * math.pow(1 + monthly_rate, tenure_months) / \
              (math.pow(1 + monthly_rate, tenure_months) - 1)
        
        return round(emi, 2)
    
    def _calculate_effective_rate(self, loan_data: Dict[str, Any]) -> float:
        """Calculate effective interest rate including fees"""
        nominal_rate = loan_data.get("interest_rate", 0)
        principal = loan_data.get("principal_amount", 0)
        
        if not principal:
            return nominal_rate
        
        # Calculate fees as percentage of principal
        fees = loan_data.get("fees", [])
        total_fees = sum(fee.get("amount", 0) for fee in fees)
        processing_fee = loan_data.get("processing_fee", 0)
        total_fees += processing_fee
        
        fees_percentage = (total_fees / principal) * 100 if principal > 0 else 0
        
        # Effective rate = nominal rate + fees percentage
        effective_rate = nominal_rate + fees_percentage
        
        return round(effective_rate, 2)
    
    def _calculate_flexibility_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate flexibility score (0-10)"""
        score = 0.0
        
        # Moratorium period adds flexibility
        moratorium = loan_data.get("moratorium_period_months", 0)
        if moratorium > 0:
            score += min(moratorium / 6 * 2, 2.0)  # Max 2 points
        
        # Prepayment options
        prepayment_penalty = loan_data.get("prepayment_penalty", "")
        if prepayment_penalty:
            if "no penalty" in prepayment_penalty.lower() or "nil" in prepayment_penalty.lower():
                score += 3.0
            elif "%" in prepayment_penalty:
                # Extract percentage
                try:
                    penalty_pct = float(prepayment_penalty.split("%")[0].split()[-1])
                    if penalty_pct < 2:
                        score += 2.0
                    elif penalty_pct < 5:
                        score += 1.0
                except:
                    score += 1.0
            else:
                score += 1.0
        else:
            score += 3.0  # Assume no penalty if not mentioned
        
        # Repayment mode flexibility
        repayment_mode = loan_data.get("repayment_mode", "")
        if repayment_mode:
            modes = repayment_mode.lower()
            if "flexible" in modes or "step" in modes:
                score += 2.0
            elif "emi" in modes:
                score += 1.0
        else:
            score += 1.0
        
        # Disbursement terms
        disbursement = loan_data.get("disbursement_terms", "")
        if disbursement and ("flexible" in disbursement.lower() or "partial" in disbursement.lower()):
            score += 2.0
        else:
            score += 1.0
        
        # Cap at 10
        score = min(score, 10.0)
        
        return round(score, 1)
    
    def compare_loans(self, loans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple loans and generate comparison result
        
        Args:
            loans: List of loan data with metrics
            
        Returns:
            Comparison result with rankings and pros/cons
        """
        try:
            if not loans:
                return {"error": "No loans to compare"}
            
            # Calculate metrics for each loan if not present
            loans_with_metrics = []
            for loan in loans:
                if "comparison_metrics" not in loan:
                    metrics = self.calculate_metrics(loan)
                    loan["comparison_metrics"] = metrics
                loans_with_metrics.append(loan)
            
            # Find best by cost
            best_by_cost = min(loans_with_metrics, 
                             key=lambda x: x.get("comparison_metrics", {}).get("total_cost_estimate", float('inf')))
            
            # Find best by flexibility
            best_by_flexibility = max(loans_with_metrics,
                                    key=lambda x: x.get("comparison_metrics", {}).get("flexibility_score", 0))
            
            # Generate pros and cons for each loan
            comparison_notes = {}
            for loan in loans_with_metrics:
                loan_id = loan.get("loan_id")
                notes = self._generate_pros_cons(loan, loans_with_metrics)
                comparison_notes[loan_id] = notes
            
            result = {
                "loans": loans_with_metrics,
                "best_by_cost": best_by_cost.get("loan_id"),
                "best_by_flexibility": best_by_flexibility.get("loan_id"),
                "comparison_notes": comparison_notes,
                "summary": {
                    "total_loans": len(loans_with_metrics),
                    "lowest_cost": best_by_cost.get("comparison_metrics", {}).get("total_cost_estimate"),
                    "highest_flexibility": best_by_flexibility.get("comparison_metrics", {}).get("flexibility_score")
                }
            }
            
            logger.info(f"Compared {len(loans)} loans")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing loans: {str(e)}")
            return {"error": str(e)}
    
    def _generate_pros_cons(self, loan: Dict[str, Any], all_loans: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate pros and cons for a loan"""
        pros = []
        cons = []
        
        metrics = loan.get("comparison_metrics", {})
        
        # Cost comparison
        total_cost = metrics.get("total_cost_estimate", 0)
        avg_cost = sum(l.get("comparison_metrics", {}).get("total_cost_estimate", 0) for l in all_loans) / len(all_loans)
        
        if total_cost < avg_cost * 0.95:
            pros.append(f"Low total cost: ${total_cost:,.2f}")
        elif total_cost > avg_cost * 1.05:
            cons.append(f"High total cost: ${total_cost:,.2f}")
        
        # Interest rate
        interest_rate = loan.get("interest_rate", 0)
        avg_rate = sum(l.get("interest_rate", 0) for l in all_loans) / len(all_loans)
        
        if interest_rate < avg_rate:
            pros.append(f"Competitive interest rate: {interest_rate}%")
        elif interest_rate > avg_rate * 1.1:
            cons.append(f"High interest rate: {interest_rate}%")
        
        # Flexibility
        flexibility = metrics.get("flexibility_score", 0)
        if flexibility >= 8.0:
            pros.append(f"High flexibility: {flexibility}/10")
        elif flexibility < 5.0:
            cons.append(f"Limited flexibility: {flexibility}/10")
        
        # Moratorium
        moratorium = loan.get("moratorium_period_months", 0)
        if moratorium > 0:
            pros.append(f"Moratorium period: {moratorium} months")
        
        # Prepayment
        prepayment = loan.get("prepayment_penalty", "")
        if prepayment and ("no penalty" in prepayment.lower() or "nil" in prepayment.lower()):
            pros.append("No prepayment penalty")
        elif prepayment and "%" in prepayment:
            cons.append(f"Prepayment penalty: {prepayment}")
        
        # EMI
        emi = metrics.get("monthly_emi", 0)
        if emi > 0:
            avg_emi = sum(l.get("comparison_metrics", {}).get("monthly_emi", 0) for l in all_loans) / len(all_loans)
            if emi < avg_emi * 0.95:
                pros.append(f"Lower monthly EMI: ${emi:,.2f}")
            elif emi > avg_emi * 1.05:
                cons.append(f"Higher monthly EMI: ${emi:,.2f}")
        
        return {
            "pros": pros if pros else ["Standard loan terms"],
            "cons": cons if cons else ["No significant drawbacks"]
        }


# Singleton instance
_calculator = None

def get_comparison_calculator() -> ComparisonCalculator:
    """Get singleton comparison calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = ComparisonCalculator()
    return _calculator

# Alias for compatibility with comparison_service
ComparisonMetricsCalculator = ComparisonCalculator