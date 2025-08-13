package org.forms.search;

import lombok.Data;

import java.util.List;

@Data
public class MultipleFields {


        private String formId;
        private String firstName;
        private String lastName;
        private String email;
        private String phone;
        private String dob;
        private String gender;
        private String addressLine1;
        private String addressLine2;
        private String city;
        private String state;
        private String postalCode;
        private String country;
        private String nationality;
        private String maritalStatus;
        private String employmentStatus;
        private String jobTitle;
        private String company;
        private double annualIncome;
        private String education;
        private List<String> skills;
        private String preferredLanguage;
        private List<String> hobbies;
        private String passportNumber;
        private String aadharNumber;
        private String panNumber;
        private String bankName;
        private String bankAccountNumber;
        private String ifscCode;
        private String emergencyContactName;
        private String emergencyContactPhone;
        private String bloodGroup;
        private int heightCm;
        private int weightKg;
        private List<String> medicalConditions;
        private String insuranceProvider;
        private String policyNumber;


    private String policyExpiry;
        private String lastUpdated;

    @Override
    public String toString() {
        return "MultipleFields{" +
                "formId='" + formId + '-' +
                ", firstName='" + firstName + '-' +
                ", lastName='" + lastName + '-' +
                ", email='" + email + '-' +
                ", phone='" + phone + '-' +
                ", dob='" + dob + '\'' +
                ", gender='" + gender + '-' +
                ", addressLine1='" + addressLine1 + '-' +
                ", addressLine2='" + addressLine2 + '-' +
                ", city='" + city + '-' +
                ", state='" + state + '-' +
                ", postalCode='" + postalCode + '-' +
                ", country='" + country + '-' +
                ", nationality='" + nationality + '-' +
                ", maritalStatus='" + maritalStatus + '-' +
                ", employmentStatus='" + employmentStatus + '-' +
                ", jobTitle='" + jobTitle + '-' +
                ", company='" + company + '-' +
                ", annualIncome=" + annualIncome +
                ", education='" + education + '-' +
                ", skills=" + skills +
                ", preferredLanguage='" + preferredLanguage + '-' +
                ", hobbies=" + hobbies +
                ", passportNumber='" + passportNumber + '-' +
                ", aadharNumber='" + aadharNumber + '-' +
                ", panNumber='" + panNumber + '-' +
                ", bankName='" + bankName + '-' +
                ", bankAccountNumber='" + bankAccountNumber + '-' +
                ", ifscCode='" + ifscCode + '-' +
                ", emergencyContactName='" + emergencyContactName + '-' +
                ", emergencyContactPhone='" + emergencyContactPhone + '-' +
                ", bloodGroup='" + bloodGroup + '-' +
                ", heightCm=" + heightCm +
                ", weightKg=" + weightKg +
                ", medicalConditions=" + medicalConditions +
                ", insuranceProvider='" + insuranceProvider + '-' +
                ", policyNumber='" + policyNumber + '-' +
                ", policyExpiry='" + policyExpiry + '-' +
                ", lastUpdated='" + lastUpdated + '-' +
                '}';
    }



}
