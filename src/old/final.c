/*
Names:	Clark LaChance
	Zach Hamilton

File Name: final_project.c

Description: Implements a program to read a file representing a car park, allows user to reserve spots, display the park, print % occupancy,
		and see closest available spot. Utilizes a nearest neighbor searching algorithm to find closest spots. Updates data set
		based on user input.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 10
#define COLS 8
char carpark[ROWS][COLS];

// Function definitions
void display();
void statistics();
int reserve();
void assign_slot(int, int);

// "D", Displays the current layout of the park. Changes based on user reservations
void display() {
	
	printf("\n   ABCD EFGH"); // Print column Labels
	
	for (int i = 0; i < ROWS; i++) { // Iterate over each row
	
		if (i % 2 == 0) { // Add a space for every other row
		
			printf("\n");
			
		}

        	for (int j = 0; j < COLS; j++) { // Iterate over each column
        	
			if (j == 0) { // For the first column of output print the row label

				printf(" %d",i+1);

			}
			
			if (j % 4 == 0){ // Add the blank driveway column as the 5th column

				printf(" ");

			}

		    	printf("%c", carpark[i][j]);// Print the current spot occupancy

        	}
        	
	printf("\n");
	
	}

}

// "S", Computes and displays occupancy percentage,
void statistics() {

	int count = 0; // Used to count the number of open spaces
	float occupancy = 0.0; // Spaces available as a percentage of the total
	
	for (int i = 0; i < COLS; i++) { // Iterate over each column
	
		for (int j = 0; j < ROWS; j++) { // Iterate over each row
		
			if (carpark[i][j] == 'O'){ // Count each spot that is a 'O' and therefore open
			
				count++;
				
			}
			
		}
		
	}
	
	occupancy = (float)count / (80) * 100; // Calculate occupancy as a percentage

	printf("Current occupancy is %.1lf%%\n", occupancy); // Final occupancy percent print

}

// "R", Finds closest available spot, returns 1 if assigned, 0 if not
int reserve() {

	int row_index = 0; // The row index of the found spot
	int col_index = 0; // The column index of the found spot

	int found = 0; // Keeps track if spot found or not
	char found_col = 'A'; // Used to display the column of the found parking spot
	int found_row = 0; // Used to display the row of the found parking spot

        char choice[2]; // User choice, 'y' or 'n'

	// Nearest neighbor search
	// i is used as the greatest column or row being searched, j iterates along that row/column
	for (int i = 0; i < COLS && !found; i++) { // Iterate over each row
	
		for (int j = 0; j < i+1 && !found; j++) { // Iterate over each column
			
			// Check the current row and index of that row
			if (carpark[i][j] == 'O') {
			
				found_col = 'A' + j; // Calculate column for print
				found_row = i + 1; // Calculate row for print
				
				col_index = j; // Save the found column
				row_index = i; // Save the found row
				
				found = 1; // Ends loop

			}

			// Check the current column and index of that column
			if (carpark[j][i] == 'O'){
			
				found_col = 'A' + i; // Calculate column for print
				found_row = j + 1; // Calculate row for print
				
				col_index = i; // Save the found column
				row_index = j; // Save the found row

				found = 1; // Ends loop
				
			}
			
		}
		
	}


	// Tests if empty spot if found
	if (found) {
	
		printf("\nClosest available slot is %c%d, do you want to reserve it (y/n)? ", found_col, found_row); // Ask if the closest spot should be chosen

		scanf("%[^\n]%*c", choice);

	// If no spot was found skip question and go to manual spot input
	} else {

		strcpy(choice, "n");
		
	}

	// If user chooses yes, reserves slot
	if (strcmp(choice, "y") == 0) {
		
		printf("Your selection %c%d is reserved!\n", found_col, found_row);

		assign_slot(row_index, col_index);

		return 1;

	// User chooses no or no spot found
	} else if (strcmp(choice, "n") == 0) {
		
		int col; // Column index calculated from col_letter
   		int row; // Requested row

    		char col_letter; // Requested column in letter form

		char location[4] = ""; // Stores user input

		printf("Please enter your preferred empty slot: "); // User input for preferred spot

    		fgets(location, sizeof(location), stdin);

    		col_letter = location[0];
		col = (int)(col_letter - 'A'); // Calculate given column index from the letter

    		row = location[1] - '0'; // Converts ASCII 0-9 to correct int
		row--;

		fflush(stdin); // Clears input buffer

		// Handles row 10, tests if third char is a zero
		if (location[2] == '0') {

			row = 9;

		}	
		
		// Verifies spot not taken and input valid
		if (col_letter > 72 || col_letter < 65 || location[1] > 57 || location[1] < 48) {

			printf("Invalid input! Exiting..\n");
	
			return 0;

		}

		if (carpark[row][col] == 'X') {

			printf("\nThat spot is already taken!\n");

			return 0;

		} else {

    			assign_slot(row, col); // Assigns the parking spot

			printf("\nYour selection %c%d is reserved!\n", col_letter, row+1);

			return 1;

		}



	} else {

		printf("Invalid input, exiting..\n"); // Exit if invalid input

		return 0;

	}


	
	return 0;

}

// Takes in row and index values and assigns the spot in carpark
void assign_slot(int row, int column) {

	carpark[row][column] = 'X';

}


int main() {

	FILE *park_dataset = fopen("carpark.txt", "r"); // Opens file read only

	// Verifies file exists
	if (park_dataset == NULL) {

		printf("Error reading file.\n");

		return -1;

	}


	// Create carpark array from data file
	for (int i = 0; i < ROWS; i++) {
	
		for (int j = 0; j < COLS; j++) {
		
			fscanf(park_dataset, " %c", &carpark[i][j]);
		}
		
	}
	
	fclose(park_dataset); // Closes file



	char user_input[1] = ""; // Stores initial user input


	// While loop that keeps program running provided the user doesn't input 'Q' to quit
	while (strcmp(user_input, "Q") != 0) { // Uses strcmp to check equality between user input and "Q"

		// Menu
		printf("\nSelect your choice from the menu:\n");
		printf("D to display current status of the car park\n");
		printf("R to reserve an empty slot in the car park\n");
		printf("S to display %% occupancy of the car park\n");
		printf("Q to quit\n");
		printf(":");

		scanf(" %[^\n]%*c", user_input);

		// User chooses "D" to display
		if (strcmp(user_input, "D") == 0) {

			display(park_dataset);


		// User chooses "R" to reserve
		} else if (strcmp(user_input, "R") == 0) {

			reserve();
			
			FILE *park_dataset = fopen("carpark.txt", "w"); // Reopens file to write new contents

			for (int i = 0; i < ROWS; i++) {

				if (i % 2 == 0 && i > 1){ // Add a space for every other row

					fprintf(park_dataset,"\n");

				}

				for (int j = 0; j < COLS; j++) {

					fprintf(park_dataset, "%c", carpark[i][j]);
					
					if (j % 3 == 0 && j > 1 && j < 5) { // Add the blank driveway column as the 5th column
					
						fprintf(park_dataset, " ");
						
					}
					
				}
				
				fprintf(park_dataset, "\n");
				
			}

			fclose(park_dataset);


		// User chooses "S" to display % occupancy
		} else if (strcmp(user_input, "S") == 0) {

			statistics();


		// User chooses "Q" to quit, or an invalid input
		} else {

			if (strcmp(user_input, "Q") != 0) {

				printf("\nInvalid input. Please enter D, R, S, or Q\n");

			}

		}
	
	}

	if (strcmp(user_input, "Q") == 0) {

		printf("Goodbye!\n");

	}

	return 0;

}
