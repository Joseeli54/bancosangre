import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './home.component';
import { OptionListComponent } from './option-list/option-list.component';
import { ProfileComponent } from './profile/profile.component';
import { SangreComponent } from './sangre/sangre.component';
import { DiagnosisComponent } from './diagnosis/diagnosis.component';
import { StatisticComponent } from './statistic/statistic.component';

const routes: Routes = [
  {
    path: '', 
    component: HomeComponent,
    children: [
      { component: OptionListComponent, path: 'option-list' },
      { component: ProfileComponent, path: 'account' },
      { component: SangreComponent, path: 'sangre' },
      { component: DiagnosisComponent, path: 'diagnostico' },
      { component: StatisticComponent, path: 'estadistica' },
    ]
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class HomeRoutingModule { }
