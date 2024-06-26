import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

const routes: Routes = [
//{path: '', redirectTo: 'trackingGeneral', pathMatch: 'full'},
{path: '', redirectTo: 'login', pathMatch: 'full'},
{path: 'home', loadChildren: () => import('./home/home.module').then(m => m.HomeModule) },
{path: 'login', loadChildren: () => import('./login/login.module').then(m => m.LoginModule) },
{path: 'register', loadChildren: () => import('./register/register.module').then(m => m.RegisterModule)},
]
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
